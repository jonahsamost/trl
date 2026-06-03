# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import aiohttp
import numpy as np
import requests
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)
from trl.import_utils import is_vllm_available
from trl.trainer.utils import print_prompt_completions_sample


if is_vllm_available(min_version="0.17.1"):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port


def _module_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _module_logger(__name__)

Messages: TypeAlias = list[dict[str, str]]


def _hash_prompt(prompt: Messages) -> str:
    """Deterministic hash for a chat-style prompt (list of message dicts)."""
    return hashlib.md5(json.dumps(prompt, sort_keys=True).encode()).hexdigest()


class PromptPassRateTracker:
    """Tracks per-prompt mean reward across epochs and retires mastered prompts.

    Stores raw reward values (not binarized) so it works naturally with both
    binary rewards (±1 for math verification) and continuous rewards (LLM-as-a-judge).
    A prompt is retired when its mean reward meets or exceeds ``threshold`` and at
    least ``min_samples`` reward observations have been recorded.

    Thread-safe: all mutations go through a lock so the tracker can be shared
    between the generate loop (which calls ``should_skip``) and the score loop
    (which calls ``update``).
    """

    def __init__(self, threshold: float = 0.9, min_samples: int = 16):
        self.threshold = threshold
        self.min_samples = min_samples
        self._history: dict[str, list[float]] = {}
        self._retired: set[str] = set()
        self._lock = threading.Lock()

    def update(self, prompt_hash: str, rewards: list[float]) -> None:
        """Record reward observations for a prompt and retire if mastered."""
        with self._lock:
            if prompt_hash in self._retired:
                return
            self._history.setdefault(prompt_hash, []).extend(rewards)
            history = self._history[prompt_hash]
            if len(history) >= self.min_samples:
                mean_reward = sum(history) / len(history)
                if mean_reward >= self.threshold:
                    self._retired.add(prompt_hash)
                    logger.info(
                        "Retired prompt %s (mean_reward=%.3f, n=%d)",
                        prompt_hash[:12], mean_reward, len(history),
                    )

    def should_skip(self, prompt_hash: str) -> bool:
        with self._lock:
            return prompt_hash in self._retired

    @property
    def num_retired(self) -> int:
        with self._lock:
            return len(self._retired)

    @property
    def num_tracked(self) -> int:
        with self._lock:
            return len(self._history)

    def save(self, path: str | Path) -> None:
        with self._lock:
            data = {
                "threshold": self.threshold,
                "min_samples": self.min_samples,
                "history": self._history,
                "retired": sorted(self._retired),
            }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> "PromptPassRateTracker":
        data = json.loads(Path(path).read_text())
        tracker = cls(
            threshold=overrides.get("threshold", data["threshold"]),
            min_samples=overrides.get("min_samples", data["min_samples"]),
        )
        tracker._history = data["history"]
        tracker._retired = set(data["retired"])
        return tracker


@dataclass(slots=True)
class RolloutGroup:
    """Single GRPO group for one prompt with multiple completions."""

    prompt: Messages
    prompt_ids: list[int]
    reward_kwargs: dict[str, list[Any]]
    completions: list[Messages]
    completions_ids: list[list[int]]
    completions_logprobs: list[list[float]]
    tool_mask: list[list[int]]
    tool_call_counts: list[int]
    tool_failure_counts: list[int]
    model_version: int
    prompt_group_id: int = 0
    prompt_hash: str = ""
    queued_at: float = 0.0


@dataclass(slots=True)
class RolloutSample:
    prompt: Messages
    completion: Messages
    input_ids: list[int]
    completion_mask: list[int]
    old_log_probs: list[float]
    advantage: float
    prompt_total_tokens: int  # sum of completion tokens across all G generations for this prompt
    prompt_group_id: int  # unique identifier for the prompt group (used by collator to count global prompts)
    model_version: int
    metrics: dict[str, float]  # logging metadata only, not used in loss computation


class VersionTrackingQueue(queue.PriorityQueue):
    """Priority queue that keeps rollout samples ordered by model version.

    The queue reports version counts through callbacks so the rollout worker can
    distinguish pending generation/scoring work from samples that are ready for
    the trainer to consume.
    """

    def __init__(
        self,
        maxsize: int,
        on_put: Callable[[int], None],
        on_get: Callable[[int], None],
    ):
        super().__init__(maxsize=maxsize)
        self._on_put = on_put
        self._on_get = on_get
        self._sequence = 0

    def _put(self, item: RolloutSample) -> None:
        sequence = self._sequence
        self._sequence += 1
        super()._put((item.model_version, sequence, item))
        self._on_put(item.model_version)

    def _get(self) -> RolloutSample:
        _, _, item = super()._get()
        self._on_get(item.model_version)
        return item


class AsyncRolloutWorker:
    """
    Minimal asynchronous actor worker structure.

    Loop:
        generate groups -> score groups -> push samples -> repeat
    """

    def __init__(
        self,
        model_name: str,
        dataset: Dataset,
        reward_funcs: list[Callable[..., list[float]]],
        tools: list[Callable] | None = None,
        environment_factory: Callable[[], object] | None = None,
        num_generations: int = 8,
        max_inflight_tasks: int = 128,
        queue_maxsize: int = 0,
        vllm_server_urls: list[str] | str = "http://localhost:8000",
        max_tokens: int = 32,
        temperature: float = 1.0,
        request_timeout: int = 120,
        server_timeout: float = 240.0,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tool_calling_iterations: int | None = None,
        log_completions: bool = False,
        num_completions_to_print: int = 3,
        max_staleness: int = 8,
        samples_per_step: int = 1,
        rollout_group_timeout: float | None = None,
        weight_names: list[str] | None = None,
        weight_dtype_names: list[str] | None = None,
        weight_shapes: list[list[int]] | None = None,
        use_lora: bool = False,
        lora_name: str | None = None,
        filter_zero_variance: bool = True,
        no_positive_resample: bool = True,
        no_positive_resample_threshold: float = 0.9,
        no_positive_resample_min_samples: int = 16,
    ):
        logger.info("initing async rollout worker")
        if not is_vllm_available(min_version="0.17.1"):
            raise ImportError(
                "vLLM >= 0.17.1 is required to use AsyncRolloutWorker. Install it with: pip install 'vllm>=0.17.1'"
            )
        self.lora_sync = use_lora
        self.filter_zero_variance = filter_zero_variance
        self.no_positive_resample = no_positive_resample
        self.pass_rate_tracker: PromptPassRateTracker | None = (
            PromptPassRateTracker(
                threshold=no_positive_resample_threshold,
                min_samples=no_positive_resample_min_samples,
            )
            if no_positive_resample
            else None
        )
        self.max_tool_calling_iterations = max_tool_calling_iterations
        self.dataset = dataset
        self._dataset_iter = iter(dataset)
        self.max_staleness = max_staleness
        self.samples_per_step = max(samples_per_step, 1)
        self.rollout_group_timeout = rollout_group_timeout
        self._version_condition = threading.Condition()
        self._live_version_counts: Counter[int] = Counter()
        self._buffered_version_counts: Counter[int] = Counter()
        self._last_capacity_log = 0.0
        self.rollout_buffer: queue.Queue[RolloutSample] = VersionTrackingQueue(
            maxsize=queue_maxsize,
            on_put=self._register_buffered_sample,
            on_get=self._unregister_buffered_sample,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._weight_update_info = {
            "names": weight_names,
            "dtype_names": weight_dtype_names,
            "shapes": weight_shapes,
            "packed": True,
            "is_checkpoint_format": True,
        }

        # When LoRA sync is active, generation requests use the LoRA adapter name
        # (e.g. "sft") while the tokenizer still loads from model_name (adapter dir).
        self.model_name = lora_name if self.lora_sync else model_name

        self.reward_funcs = reward_funcs
        self.reward_func_names = [f.__name__ for f in reward_funcs]
        self.num_generations = num_generations
        self.max_inflight_tasks = max_inflight_tasks
        self.environments = None
        environment_methods = [[] for _ in range(self.max_inflight_tasks)]
        if environment_factory is not None:
            self.environments = [environment_factory() for _ in range(self.max_inflight_tasks)]
            for i, environment in enumerate(self.environments):
                has_reset = False
                for name, member in inspect.getmembers(environment, predicate=inspect.ismethod):
                    if name == "reset":
                        has_reset = True
                    elif not name.startswith("_"):
                        environment_methods[i].append(member)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define `reset`."
                    )

        base_tools = tools or []
        self._sync_tool_dicts = [{} for _ in range(self.max_inflight_tasks)]
        for i in range(self.max_inflight_tasks):
            for tool in base_tools + (environment_methods[i] if self.environments is not None else []):
                if inspect.iscoroutinefunction(tool):
                    raise ValueError("Asynchronous tools are not supported in AsyncRolloutWorker yet.")
                self._sync_tool_dicts[i][tool.__name__] = tool
        self.tools = base_tools + (environment_methods[0] if self.environments is not None else [])

        if isinstance(vllm_server_urls, str):
            self.vllm_server_urls = [vllm_server_urls.rstrip("/")]
        else:
            self.vllm_server_urls = [u.rstrip("/") for u in vllm_server_urls]
        self.vllm_server_url = self.vllm_server_urls[0]
        self.model_update_groups: list = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.server_timeout = server_timeout
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.log_completions = log_completions
        self.num_completions_to_print = num_completions_to_print
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Always use original path for tokenizer
        self.tokenizer = add_response_schema(self.tokenizer)
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools and not is_chat_template_prefix_preserving(self.tokenizer):
            self.chat_template = get_training_chat_template(self.tokenizer)
        else:
            self.chat_template = None

        self._groups_to_score: asyncio.Queue[RolloutGroup | None] = asyncio.Queue(maxsize=16)
        self._total_completion_tokens = 0
        self._total_groups_scored = 0
        self._group_metrics_lock = threading.Lock()
        self._groups_seen_total = 0
        self._groups_trainable_total = 0
        self._groups_zero_variance_dropped_total = 0
        self._generation_start_time: float | None = None
        self.model_version = 0
        self.session = None

        self._lora_nccl_groups: list = []
        self._wait_for_server_ready_sync(timeout_s=self.server_timeout)
        if self.lora_sync:
            logger.info("LoRA sync mode: initializing direct NCCL LoRA transfer for %d server(s)", len(self.vllm_server_urls))
            for url in self.vllm_server_urls:
                self._init_lora_sync_group_for(url)
        else:
            for url in self.vllm_server_urls:
                self._init_weight_transfer_for(url)

    def _record_group_outcome(self, outcome: str) -> None:
        """Track group-level outcomes that may not produce trainable samples."""
        with self._group_metrics_lock:
            self._groups_seen_total += 1
            if outcome == "trainable":
                self._groups_trainable_total += 1
            elif outcome == "zero_variance":
                self._groups_zero_variance_dropped_total += 1
            else:
                raise ValueError(f"Unknown group outcome: {outcome}")

    def _group_metrics_snapshot(self) -> dict[str, float]:
        """Return cumulative group outcome counters for W&B logging."""
        with self._group_metrics_lock:
            groups_seen = max(self._groups_seen_total, 1)
            return {
                "groups_seen_total": float(self._groups_seen_total),
                "groups_trainable_total": float(self._groups_trainable_total),
                "groups_dropped_total": float(self._groups_zero_variance_dropped_total),
                "groups_zero_variance_dropped_total": float(self._groups_zero_variance_dropped_total),
                "groups_zero_variance_drop_rate": self._groups_zero_variance_dropped_total / groups_seen,
                "groups_drop_rate": self._groups_zero_variance_dropped_total / groups_seen,
            }

    def _wait_for_server_ready_sync(self, timeout_s: float = 240.0, poll_interval_s: float = 2.0) -> None:
        """Block until all vLLM servers are healthy."""
        for url in self.vllm_server_urls:
            self._wait_for_one_server(url, timeout_s, poll_interval_s)

    @staticmethod
    def _wait_for_one_server(url: str, timeout_s: float = 240.0, poll_interval_s: float = 2.0) -> None:
        logger.info(f"Waiting for vLLM server at {url} ...")
        start = time.time()
        while True:
            elapsed = time.time() - start
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server at {url} ready after {elapsed:.1f}s")
                    return
            except (requests.ConnectionError, requests.Timeout, OSError):
                pass
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s waiting for vLLM server at {url}. "
                    "Make sure the vLLM server is running and reachable. If the server needs more time to load "
                    "the model, increase `vllm_server_timeout` in your AsyncGRPOConfig."
                )
            if int(elapsed) % 10 < poll_interval_s:
                logger.info(f"Still waiting for vLLM server at {url}... ({elapsed:.0f}s)")
            time.sleep(poll_interval_s)

    def _init_weight_transfer_for(self, url: str) -> None:
        response = requests.get(f"{url}/get_world_size")
        inference_world_size = response.json()["world_size"]
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()

        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{url}/init_weight_transfer_engine",),
            kwargs={"json": {"init_info": init_info}, "timeout": 120},
        )
        t_init.start()
        group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        t_init.join()
        self.model_update_groups.append(group)
        logger.info("Init weight sync group with vLLM at %s", url)

    def _init_lora_sync_group_for(self, url: str) -> None:
        """Initialize a dedicated NCCL group for LoRA-only weight transfer to one server."""
        response = requests.get(f"{url}/get_world_size")
        inference_world_size = response.json()["world_size"]
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()

        init_info = {
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": world_size,
            }
        }
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{url}/init_lora_sync_group",),
            kwargs={"json": init_info, "timeout": 120},
        )
        t_init.start()

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        from src.h3_rl.train.src.rl_trl.lora_worker_extension import _create_stateless_pg

        pg = _create_stateless_pg(
            host=master_address, port=master_port, rank=0, world_size=world_size
        )
        nccl_group = PyNcclCommunicator(pg, device=torch.device(f"cuda:{torch.cuda.current_device()}"))
        t_init.join()
        self._lora_nccl_groups.append(nccl_group)
        logger.info("LoRA NCCL sync group initialized for %s (world_size=%d)", url, world_size)

    def send_lora_weights(self, lora_param_iter, lora_alpha: float, lora_rank: int, lora_int_id: int) -> None:
        """Send LoRA A/B tensors to all vLLM servers via NCCL.

        Args:
            lora_param_iter: Iterator of (peft_param_name, tensor) for LoRA params only.
            lora_alpha: The LoRA alpha scaling factor from the adapter config.
            lora_rank: The LoRA rank from the adapter config.
            lora_int_id: The integer ID of the LoRA adapter in vLLM's slot table.
        """
        if not self._lora_nccl_groups:
            logger.warning("LoRA NCCL groups not initialized, skipping send_lora_weights")
            return

        t0 = time.time()

        params = []
        tensors = []
        for name, tensor in lora_param_iter:
            params.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).split(".")[-1],
            })
            tensors.append(tensor)

        manifest = json.dumps({
            "lora_alpha": lora_alpha,
            "lora_rank": lora_rank,
            "lora_int_id": lora_int_id,
            "params": params,
        })

        import torch as _torch
        stream = _torch.cuda.current_stream()

        for url, nccl_group in zip(self.vllm_server_urls, self._lora_nccl_groups, strict=True):
            t_update = threading.Thread(
                target=requests.post,
                args=(f"{url}/update_lora",),
                kwargs={"json": {"manifest_json": manifest}, "timeout": 300},
            )
            t_update.start()
            for tensor in tensors:
                nccl_group.broadcast(tensor, src=0, stream=stream)
            t_update.join()

        logger.info(
            "[weight_sync] LoRA NCCL send to %d server(s) took %.1fs (%d params)",
            len(self.vllm_server_urls), time.time() - t0, len(tensors),
        )

    def update_model_version(self, model_version: int):
        self.model_version = model_version
        with self._version_condition:
            self._version_condition.notify_all()

    @staticmethod
    def _oldest_version(counts: Counter[int]) -> int | None:
        return min((version for version, count in counts.items() if count > 0), default=None)

    def _increment_version_count(self, counts: Counter[int], version: int, count: int) -> None:
        if count <= 0:
            return
        counts[version] += count

    def _decrement_version_count(self, counts: Counter[int], version: int, count: int) -> None:
        if count <= 0:
            return
        counts[version] -= count
        if counts[version] <= 0:
            del counts[version]

    def _register_pending_group(self, version: int) -> None:
        with self._version_condition:
            self._increment_version_count(self._live_version_counts, version, self.num_generations)
            self._version_condition.notify_all()

    def _unregister_pending_group(self, version: int) -> None:
        with self._version_condition:
            self._decrement_version_count(self._live_version_counts, version, self.num_generations)
            self._version_condition.notify_all()

    def _register_buffered_sample(self, version: int) -> None:
        with self._version_condition:
            self._increment_version_count(self._live_version_counts, version, 1)
            self._increment_version_count(self._buffered_version_counts, version, 1)
            self._version_condition.notify_all()

    def _unregister_buffered_sample(self, version: int) -> None:
        with self._version_condition:
            self._decrement_version_count(self._live_version_counts, version, 1)
            self._decrement_version_count(self._buffered_version_counts, version, 1)
            self._version_condition.notify_all()

    def _get_admissible_generation_version(self) -> int | None:
        """Return a version that can admit one group, or ``None`` if capacity is full."""
        with self._version_condition:
            version = self.model_version
            live_count = self._live_version_counts.get(version, 0)
            max_live_for_version = (self.max_staleness + 1) * self.samples_per_step
            oldest_live = self._oldest_version(self._live_version_counts)
            oldest_staleness = version - oldest_live if oldest_live is not None else None
            if self.num_generations > max_live_for_version:
                raise ValueError(
                    "Staleness window is too small for one prompt group "
                    f"(num_generations={self.num_generations}, max_live_for_version={max_live_for_version}, "
                    f"samples_per_step={self.samples_per_step}, max_staleness={self.max_staleness})."
                )
            if oldest_staleness is None and live_count + self.num_generations <= max_live_for_version:
                return version
            if (
                oldest_staleness is not None
                and oldest_staleness < self.max_staleness
                and live_count + self.num_generations <= max_live_for_version
            ):
                return version

        now = time.monotonic()
        if now - self._last_capacity_log > 5:
            if oldest_staleness is not None and oldest_staleness >= self.max_staleness:
                logger.info(
                    "[staleness] waiting to admit rollouts for version=%d because oldest live "
                    "version=%d is at staleness=%d (max=%d)",
                    version,
                    oldest_live,
                    oldest_staleness,
                    self.max_staleness,
                )
            else:
                logger.info(
                    "[staleness] waiting to admit rollouts for version=%d "
                    "(live=%d, add=%d, max=%d)",
                    version,
                    live_count,
                    self.num_generations,
                    max_live_for_version,
                )
            self._last_capacity_log = now
        return None

    def _expire_timed_out_pending_groups(
        self,
        pending_groups: dict[int, RolloutGroup],
        pending_completed: dict[int, int],
        pending_started_at: dict[int, float],
        inflight_tasks: dict[asyncio.Task, tuple[int, int]],
        free_slots: set[int],
        abandoned_group_ids: set[int],
    ) -> list[asyncio.Task]:
        """Abandon pending groups that exceeded the configured group timeout."""
        if self.rollout_group_timeout is None:
            return []

        now = time.monotonic()
        expired_group_ids = [
            group_id
            for group_id, started_at in pending_started_at.items()
            if now - started_at > self.rollout_group_timeout
        ]
        cancelled_tasks = []
        for group_id in expired_group_ids:
            group = pending_groups.pop(group_id, None)
            if group is None:
                pending_completed.pop(group_id, None)
                pending_started_at.pop(group_id, None)
                continue

            completed = pending_completed.pop(group_id, 0)
            started_at = pending_started_at.pop(group_id)
            abandoned_group_ids.add(group_id)
            for task, (task_group_id, slot) in list(inflight_tasks.items()):
                if task_group_id != group_id:
                    continue
                del inflight_tasks[task]
                free_slots.add(slot)
                task.cancel()
                cancelled_tasks.append(task)

            self._unregister_pending_group(group.model_version)
            logger.warning(
                "[timeout] abandoning rollout group=%d version=%d completed=%d/%d age=%.1fs timeout=%.1fs",
                group_id,
                group.model_version,
                completed,
                self.num_generations,
                now - started_at,
                self.rollout_group_timeout,
            )
        return cancelled_tasks

    def wait_until_next_sample_allowed(
        self,
        current_version: int,
        max_staleness: int,
        timeout: float,
    ) -> None:
        """Block the trainer when old pending rollouts need to be consumed next.

        This is the consume-side half of the staleness cap. Generation admission
        prevents creating more samples for a version than can be trained within
        the window; this guard prevents the trainer from burning the remaining
        window on newer samples while older rollouts are still pending.
        """
        deadline = time.monotonic() + timeout
        last_log = 0.0
        with self._version_condition:
            while True:
                oldest_live = self._oldest_version(self._live_version_counts)
                if oldest_live is None:
                    return

                staleness = current_version - oldest_live
                if staleness > max_staleness:
                    return

                live_count = self._live_version_counts[oldest_live]
                remaining_steps = math.ceil(live_count / self.samples_per_step)
                must_prioritize_oldest = staleness + remaining_steps - 1 >= max_staleness
                oldest_buffered = self._oldest_version(self._buffered_version_counts)
                if not must_prioritize_oldest or oldest_buffered == oldest_live:
                    return

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for old rollouts to become trainable "
                        f"(oldest_live_version={oldest_live}, current_version={current_version}, "
                        f"staleness={staleness}, live_count={live_count}, "
                        f"samples_per_step={self.samples_per_step}, max_staleness={max_staleness})."
                    )

                now = time.monotonic()
                if now - last_log > 5:
                    logger.info(
                        "[staleness] waiting for version=%d rollouts before consuming newer samples "
                        "(current=%d, staleness=%d, live=%d, buffered_oldest=%s)",
                        oldest_live,
                        current_version,
                        staleness,
                        live_count,
                        oldest_buffered,
                    )
                    last_log = now
                self._version_condition.wait(timeout=min(0.5, remaining))

    async def _run_loops(self, stop_event: asyncio.Event) -> None:
        async with aiohttp.ClientSession() as session:
            self.session = session
            logger.info(
                f"vllm worker started: num_generations={self.num_generations}, max_inflight_tasks={self.max_inflight_tasks}"
            )
            await asyncio.gather(
                asyncio.create_task(self._generate_loop(stop_event=stop_event)),
                asyncio.create_task(self._score_loop(stop_event=stop_event)),
            )

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def stop(self) -> None:
        logger.info("Stopping worker thread...")
        with self._version_condition:
            self._version_condition.notify_all()
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                pass

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._stop_event = asyncio.Event()
        try:
            loop.run_until_complete(self._run_loops(stop_event=self._stop_event))
        except Exception as e:
            logger.exception(f"Worker thread failed: {e}")
            raise
        finally:
            loop.close()
            self._destroy_model_update_group()

    def _destroy_model_update_group(self) -> None:
        for group in self.model_update_groups:
            group.group.store = None
            group.group.socket = None
        self.model_update_groups.clear()
        self._lora_nccl_groups.clear()

    def pause(self, clear_cache: bool = False) -> None:
        t0 = time.time()
        for url in self.vllm_server_urls:
            requests.post(f"{url}/pause", params={"mode": "keep", "clear_cache": str(clear_cache).lower()})
        logger.debug(f"[weight_sync] pause HTTP for {len(self.vllm_server_urls)} server(s) took {time.time() - t0:.1f}s")

    def resume(self) -> None:
        t0 = time.time()
        for url in self.vllm_server_urls:
            requests.post(f"{url}/resume")
        logger.debug(f"[weight_sync] resume HTTP for {len(self.vllm_server_urls)} server(s) took {time.time() - t0:.1f}s")

    def reload_lora(self, adapter_path: str, lora_name: str) -> None:
        """Tell all vLLM servers to hot-reload a LoRA adapter from disk."""
        t0 = time.time()
        payload = {
            "lora_name": lora_name,
            "lora_path": adapter_path,
            "load_inplace": True,
        }
        for url in self.vllm_server_urls:
            resp = requests.post(f"{url}/v1/load_lora_adapter", json=payload, timeout=120)
            resp.raise_for_status()
        logger.info(f"[weight_sync] LoRA reload ({lora_name}) to {len(self.vllm_server_urls)} server(s) took {time.time() - t0:.1f}s")

    def send_weights(self, iterator) -> None:
        if not self.model_update_groups:
            return
        t0 = time.time()

        # Materialize the iterator once so we can replay it for each server
        weight_list = list(iterator)

        for url, group in zip(self.vllm_server_urls, self.model_update_groups, strict=True):
            t_update = threading.Thread(
                target=requests.post,
                args=(f"{url}/update_weights",),
                kwargs={"json": {"update_info": self._weight_update_info}, "timeout": 1800},
            )
            t_update.start()
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(weight_list),
                trainer_args=NCCLTrainerSendWeightsArgs(group=group, packed=True),
            )
            t_update.join()

        logger.info(
            "[weight_sync] send_weights to %d server(s) took %.1fs",
            len(self.vllm_server_urls), time.time() - t0,
        )

    async def _generate_loop(self, stop_event: asyncio.Event) -> None:
        pending_groups: dict[int, RolloutGroup] = {}
        pending_completed: dict[int, int] = {}
        pending_started_at: dict[int, float] = {}
        inflight_tasks: dict[asyncio.Task, tuple[int, int]] = {}
        abandoned_group_ids: set[int] = set()
        free_slots = set(range(self.max_inflight_tasks))
        work_iter = self._repeat_iterator()
        pending_work: tuple[int, dict[str, Any]] | None = None
        total_completions_finished = 0
        _server_rr_counter = 0

        self._generation_start_time = time.monotonic()
        try:
            while True:
                cancelled_tasks = self._expire_timed_out_pending_groups(
                    pending_groups=pending_groups,
                    pending_completed=pending_completed,
                    pending_started_at=pending_started_at,
                    inflight_tasks=inflight_tasks,
                    free_slots=free_slots,
                    abandoned_group_ids=abandoned_group_ids,
                )
                if cancelled_tasks:
                    await asyncio.gather(*cancelled_tasks, return_exceptions=True)

                while free_slots and not stop_event.is_set():
                    if pending_work is None:
                        pending_work = next(work_iter)
                    group_id, row = pending_work
                    if group_id in abandoned_group_ids:
                        pending_work = None
                        continue

                    if group_id not in pending_groups:
                        model_version = self._get_admissible_generation_version()
                        if model_version is None:
                            break
                        prompt = row["prompt"]
                        prompt_ids = self.tokenizer.apply_chat_template(
                            prompt,
                            return_dict=False,
                            add_generation_prompt=True,
                            tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
                            chat_template=self.chat_template,
                            **self.chat_template_kwargs,
                        )
                        reward_kwargs = {
                            key: [row[key]] * self.num_generations
                            for key in row
                            if key not in {"prompt", "completion", "completion_ids"}
                        }
                        pending_groups[group_id] = RolloutGroup(
                            prompt=prompt,
                            prompt_ids=prompt_ids,
                            reward_kwargs=reward_kwargs,
                            completions=[],
                            completions_ids=[],
                            completions_logprobs=[],
                            tool_mask=[],
                            tool_call_counts=[],
                            tool_failure_counts=[],
                            model_version=model_version,
                            prompt_group_id=group_id,
                            prompt_hash=_hash_prompt(prompt),
                        )
                        self._register_pending_group(model_version)
                        pending_completed[group_id] = 0
                        pending_started_at[group_id] = time.monotonic()
                        logger.debug(f"Started group {group_id}; pending_groups={len(pending_groups)}")

                    slot = free_slots.pop()
                    if self.environments is not None:
                        # Current assumption: reset side effects matter, return value is ignored.
                        self.environments[slot].reset(**row)

                    server_url = self.vllm_server_urls[_server_rr_counter % len(self.vllm_server_urls)]
                    _server_rr_counter += 1
                    logger.info(f"[slot] assigned slot={slot} group={group_id} server={server_url} free_after={len(free_slots)}")
                    task = asyncio.create_task(
                        self._generate_one(pending_groups[group_id].prompt, tool_dict=self._sync_tool_dicts[slot], server_url=server_url)
                    )
                    inflight_tasks[task] = (group_id, slot)
                    pending_work = None

                if not inflight_tasks:
                    if stop_event.is_set():
                        return
                    await asyncio.sleep(0.01)
                    continue

                done, _ = await asyncio.wait(inflight_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                if not done:
                    if not free_slots:
                        logger.debug(
                            f"[generate] all {self.max_inflight_tasks} slots busy, "
                            f"pending_groups={len(pending_groups)}, waiting for completions..."
                        )
                    continue

                for task in done:
                    group_id, slot = inflight_tasks.pop(task)
                    free_slots.add(slot)
                    logger.debug(f"[slot] freed   slot={slot} group={group_id} free_after={len(free_slots)}")
                    if task.exception() is not None:
                        raise task.exception()

                    (
                        completion,
                        completion_ids,
                        completion_logprobs,
                        tool_mask,
                        tool_call_count,
                        tool_failure_count,
                    ) = task.result()
                    group = pending_groups[group_id]
                    group.completions.append(completion)
                    group.completions_ids.append(completion_ids)
                    group.completions_logprobs.append(completion_logprobs)
                    group.tool_mask.append(tool_mask)
                    group.tool_call_counts.append(tool_call_count)
                    group.tool_failure_counts.append(tool_failure_count)
                    # TODO: move this in generation task, shouldn't matter but is correct
                    self._total_completion_tokens += sum(tool_mask)
                    pending_completed[group_id] += 1
                    total_completions_finished += 1
                    group_done = pending_completed[group_id]
                    wave_done = sum(pending_completed.values())
                    wave_expected = len(pending_groups) * self.num_generations
                    logger.info(
                        "[generate] completion finished: group=%d group_done=%d/%d "
                        "wave_done=%d/%d total_done=%d inflight=%d free_slots=%d pending_groups=%d",
                        group_id,
                        group_done,
                        self.num_generations,
                        wave_done,
                        wave_expected,
                        total_completions_finished,
                        len(inflight_tasks),
                        len(free_slots),
                        len(pending_groups),
                    )

                    if pending_completed[group_id] == self.num_generations:
                        group.queued_at = time.monotonic()
                        while True:
                            try:
                                self._groups_to_score.put_nowait(group)
                                break
                            except asyncio.QueueFull:
                                if stop_event.is_set():
                                    return
                                await asyncio.sleep(0.1)
                        logger.info(f"Group {group_id} complete; queued_for_scoring={self._groups_to_score.qsize()}")
                        del pending_groups[group_id]
                        del pending_completed[group_id]
                        del pending_started_at[group_id]
        finally:
            for task in inflight_tasks:
                task.cancel()
            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)
            for group in pending_groups.values():
                self._unregister_pending_group(group.model_version)
            # Use put_nowait: if the queue is full at shutdown, skip the sentinel —
            # _score_loop will exit via stop_event check in its outer loop.
            try:
                self._groups_to_score.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def _compute_rollout_metrics(self, samples: list[RolloutSample], scoring_time: float, wait_scoring: float) -> None:
        assert self._generation_start_time is not None, "generation_start_time init in run()"
        elapsed = time.monotonic() - self._generation_start_time
        generation_tok_per_sec = self._total_completion_tokens / elapsed if elapsed > 0 else 0.0

        scoring_time_ms = scoring_time * 1000
        wait_scoring_ms = wait_scoring * 1000

        for sample in samples:
            sample.metrics["generation_tok_per_s"] = generation_tok_per_sec
            sample.metrics["scoring_time_ms"] = scoring_time_ms
            sample.metrics["wait_scoring_ms"] = wait_scoring_ms
            sample.metrics["buffer_qsize"] = self.rollout_buffer.qsize()

        logger.info(
            f"[inference] total_completion_tokens={self._total_completion_tokens}, "
            f"generation_tok/s={generation_tok_per_sec:.1f}, scoring_time={scoring_time_ms:.1f}ms, "
            f"wait_scoring={wait_scoring_ms:.1f}ms"
        )

    async def _score_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            t_wait = time.monotonic()
            try:
                group = await asyncio.wait_for(self._groups_to_score.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if group is None:
                return
            score_queue_wait = time.monotonic() - t_wait

            wait_scoring = time.monotonic() - group.queued_at

            if score_queue_wait > 0.5:
                logger.info(f"[score] waited {score_queue_wait:.1f}s for a group to score")

            try:
                t0 = time.monotonic()
                samples = await self._score_group(group)
                scoring_time = time.monotonic() - t0
                logger.info(
                    f"[score] scored {len(samples)} samples in {scoring_time:.2f}s, "
                    f"buffer_qsize={self.rollout_buffer.qsize()}"
                )

                self._compute_rollout_metrics(samples, scoring_time, wait_scoring)

                if self.log_completions and samples:
                    print_prompt_completions_sample(
                        prompts=[s.prompt for s in samples],
                        completions=[s.completion for s in samples],
                        rewards={"reward": [s.metrics["reward"] for s in samples]},
                        advantages=[s.advantage for s in samples],
                        step=self._total_groups_scored,
                        num_samples=self.num_completions_to_print,
                    )
                self._total_groups_scored += 1

                for sample in samples:
                    logger.info('[enqueue] pushing to rollout buffer')
                    while True:
                        try:
                            self.rollout_buffer.put_nowait(sample)
                            break
                        except queue.Full:
                            if stop_event.is_set():
                                return
                            # Wait for trainer to consume loop
                            logger.info(
                                f"[score] rollout buffer full (maxsize={self.rollout_buffer.maxsize}), waiting for trainer to consume..."
                            )
                            await asyncio.sleep(0.1)

                logger.info(
                    f"Scored group with {len(samples)} samples; rollout_buffer_qsize={self.rollout_buffer.qsize()}"
                )
            finally:
                self._unregister_pending_group(group.model_version)

    def _repeat_iterator(self) -> Iterator[tuple[int, dict[str, Any]]]:
        group_id = 0
        dataset_len = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        consecutive_skips = 0
        while True:
            try:
                row = next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = iter(self.dataset)
                row = next(self._dataset_iter)

            if self.pass_rate_tracker is not None:
                prompt_hash = _hash_prompt(row["prompt"])
                if self.pass_rate_tracker.should_skip(prompt_hash):
                    consecutive_skips += 1
                    if dataset_len and consecutive_skips >= dataset_len:
                        logger.warning(
                            "All %d prompts retired (num_retired=%d). "
                            "Training data exhausted — stopping generation.",
                            dataset_len, self.pass_rate_tracker.num_retired,
                        )
                        return
                    continue
                consecutive_skips = 0

            for _ in range(self.num_generations):
                yield group_id, row
            group_id += 1

    async def _generate_one(
        self, prompt: Messages, tool_dict: dict[str, Callable], server_url: str | None = None,
    ) -> tuple[list[dict[str, str]], list[int], list[float], list[int], int, int]:
        url = server_url or self.vllm_server_url
        completion, completion_ids, completion_logprobs, tool_mask = [], [], [], []
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        max_iterations = self.max_tool_calling_iterations
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            return_dict=False,
            add_generation_prompt=True,
            tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        while True:
            turn_ids, turn_logprobs = await self._generate_one_turn(prompt_ids, server_url=url)
            assistant_message = parse_response(self.tokenizer, turn_ids)
            completion.append(assistant_message)
            completion_ids.extend(turn_ids)
            completion_logprobs.extend(turn_logprobs)
            tool_mask.extend([1] * len(turn_ids))
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None or (max_iterations is not None and iteration_num >= max_iterations):
                return completion, completion_ids, completion_logprobs, tool_mask, tool_call_count, tool_failure_count

            tool_messages, n_calls, n_failures = self._execute_tool_calls(tool_calls, tool_dict)
            tool_call_count += n_calls
            tool_failure_count += n_failures
            completion.extend(tool_messages)
            suffix_ids = self._get_tool_suffix_ids(tool_messages)
            completion_ids.extend(suffix_ids)
            completion_logprobs.extend([0.0] * len(suffix_ids))
            tool_mask.extend([0] * len(suffix_ids))
            prompt_ids = prompt_ids + turn_ids + suffix_ids
            iteration_num += 1

    def _get_tool_suffix_ids(self, tool_messages: list[dict[str, Any]]) -> list[int]:
        """Get token IDs for tool result formatting by using a minimal dummy conversation."""
        # Use the real tool name instead of a dummy: some templates (e.g. GPT-OSS) derive the tool response
        # header from the assistant's tool call name.
        dummy_tool_calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {
                "role": "assistant",
                # "content" is required here because VLM processors crash on tokenize=True without it
                # (KeyError in processing_utils.py). See huggingface/transformers#45290.
                "content": "",
                "tool_calls": dummy_tool_calls,
            },
        ]
        prefix_ids = self.tokenizer.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        full_ids = self.tokenizer.apply_chat_template(
            dummy_messages + tool_messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )

        # Some chat templates (notably Qwen3/Qwen3.5) render "...<|im_end|>\n" after an assistant/tool block.
        # When we compute `suffix_ids` by slicing `full_ids`, we must align the slicing boundary to
        # EOS (not EOS + newline). Templates that don't use EOS as end-of-turn (e.g. Gemma uses
        # <turn|>) skip this trimming.
        eos_positions = [i for i, tok_id in enumerate(prefix_ids) if tok_id == self.tokenizer.eos_token_id]
        if eos_positions:
            prefix_ids = prefix_ids[: eos_positions[-1] + 1]

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Unexpected tokenization: the EOS-trimmed prefix IDs are not a prefix of the full IDs.")

        return full_ids[len(prefix_ids) :]

    def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]], tool_dict: dict[str, Callable]
    ) -> tuple[list[dict[str, str]], int, int]:
        tool_messages = []
        n_calls = 0
        n_failures = 0
        for tool_call in tool_calls:
            n_calls += 1
            function = tool_call["function"]
            name = function["name"]
            try:
                arguments = function.get("arguments", {})
                result = tool_dict[name](**arguments)
            except Exception as error:
                n_failures += 1
                result = {"error": str(error)}
            tool_messages.append({"role": "tool", "name": name, "content": str(result)})
        return tool_messages, n_calls, n_failures

    async def _generate_one_turn(self, prompt_ids: list[int], server_url: str | None = None) -> tuple[list[int], list[float]]:
        url = server_url or self.vllm_server_url
        payload = {
            "model": self.model_name,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
        }
        while True:
            try:
                output = await self._post(url, "/v1/completions", payload, self.request_timeout)
                break
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, aiohttp.ClientResponseError):
                logger.debug("Server %s unavailable (likely weight sync pause), retrying...", url)
                await asyncio.sleep(1.0)
        choice = output["choices"][0]
        if choice.get('finish_reason', '') == 'length':
            logger.info("Completion length bounded by max_tokens")
        completion_ids = choice["token_ids"]
        completion_logprobs = choice["logprobs"]["token_logprobs"]
        return completion_ids, completion_logprobs

    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        kwargs = dict(
            completions=group.completions,
            prompt=group.prompt,
            prompts=[group.prompt] * len(group.completions),
            completion_ids=group.completions_ids,
            **group.reward_kwargs,
        )
        all_rewards = await asyncio.gather(
            *[
                reward_func(**kwargs)
                if inspect.iscoroutinefunction(reward_func)
                else asyncio.to_thread(reward_func, **kwargs)
                for reward_func in self.reward_funcs
            ]
        )

        # Sum rewards across all reward functions. Reward functions may return None for individual
        # samples (e.g. accuracy_reward when the gold solution is unparseable). Convert None → nan
        # and use nansum so that a None from one function doesn't affect the others, matching TRL.
        all_rewards = [[r if r is not None else float("nan") for r in row] for row in all_rewards]
        rewards = np.nansum(np.array(all_rewards, dtype=float), axis=0)
        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())

        if self.pass_rate_tracker is not None and group.prompt_hash:
            self.pass_rate_tracker.update(group.prompt_hash, rewards.tolist())

        if self.filter_zero_variance and reward_std < 1e-8:
            self._record_group_outcome("zero_variance")
            logger.info(
                f"Dropping zero-variance group (prompt_group_id={group.prompt_group_id}, "
                f"reward={rewards[0]:.4f}, n={len(rewards)}, group_metrics={self._group_metrics_snapshot()})"
            )
            return []

        self._record_group_outcome("trainable")
        advantages = rewards - reward_mean
        logger.info(
            "Rollout metrics: reward_mean=%.4f, reward_std=%.4f, group_metrics=%s",
            reward_mean,
            reward_std,
            self._group_metrics_snapshot(),
        )

        # tools/call_frequency: mean calls per completion (matches TRL's total_calls / num_completions)
        # tools/failure_frequency: per-completion failure rate; averaged across samples in compute_loss
        #   (TRL uses total_failures / total_calls, ours weights equally per completion — close enough)
        total_calls = sum(group.tool_call_counts)
        tool_metrics = (
            [
                {
                    "tools/call_frequency": float(n_calls),
                    "tools/failure_frequency": (n_failures / n_calls) if n_calls > 0 else 0.0,
                }
                for n_calls, n_failures in zip(group.tool_call_counts, group.tool_failure_counts, strict=True)
            ]
            if total_calls > 0
            else [{}] * len(group.completions)
        )

        per_func_rewards = np.array(all_rewards, dtype=float)  # shape (num_funcs, num_completions)
        prompt_total_tokens = sum(sum(mask) for mask in group.tool_mask)

        tracker_metrics: dict[str, float] = {}
        if self.pass_rate_tracker is not None:
            tracker_metrics["prompts_retired"] = float(self.pass_rate_tracker.num_retired)
            tracker_metrics["prompts_tracked"] = float(self.pass_rate_tracker.num_tracked)
        tracker_metrics.update(self._group_metrics_snapshot())

        return [
            RolloutSample(
                prompt=group.prompt,
                completion=completion,
                input_ids=group.prompt_ids + completion_ids,
                completion_mask=[0] * len(group.prompt_ids) + tool_mask,
                old_log_probs=[0.0] * len(group.prompt_ids) + logprobs,
                advantage=advantage,
                prompt_total_tokens=prompt_total_tokens,
                prompt_group_id=group.prompt_group_id,
                model_version=group.model_version,
                metrics={
                    "reward": float(reward),
                    "reward_std": reward_std,
                    **{
                        f"rewards/{name}": float(func_reward)
                        for name, func_reward in zip(self.reward_func_names, per_func_rewards[:, i], strict=True)
                    },
                    **tm,
                    **tracker_metrics,
                },
            )
            for i, (completion, completion_ids, logprobs, tool_mask, advantage, reward, tm) in enumerate(
                zip(
                    group.completions,
                    group.completions_ids,
                    group.completions_logprobs,
                    group.tool_mask,
                    advantages,
                    rewards,
                    tool_metrics,
                    strict=True,
                )
            )
        ]

    async def _post(self, base_url: str, path: str, payload: dict, timeout: float, max_retries: int = 3) -> dict:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{base_url}{path}", json=payload, timeout=client_timeout
                ) as response:
                    response.raise_for_status()
                    content = await response.json()
                    return content if content else {}
            except (TimeoutError, asyncio.TimeoutError):
                if attempt < max_retries - 1:
                    logger.warning(f"POST {base_url}{path} timed out (attempt {attempt + 1}/{max_retries}), retrying...")
                    await asyncio.sleep(1)
                else:
                    raise
