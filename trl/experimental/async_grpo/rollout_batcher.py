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

import logging
import queue
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from accelerate.utils import (
    broadcast,
    broadcast_object_list,
    get_data_structure,
    initialize_tensors,
    send_to_device,
    slice_tensors,
)
from torch.utils.data import IterableDataset
from transformers.data.data_collator import DataCollatorMixin

from trl.trainer.utils import pad

from .async_grpo_config import AdvantageNormalization


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


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        attention_mask = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]
        completion_mask = [torch.tensor(example["completion_mask"], dtype=torch.float32) for example in examples]
        old_log_probs = [torch.tensor(example["old_log_probs"], dtype=torch.float32) for example in examples]
        advantages = torch.tensor([example["advantage"] for example in examples], dtype=torch.float32)
        batch_adv_std = torch.tensor([example["batch_adv_std"] for example in examples], dtype=torch.float32)
        prompt_total_tokens = torch.tensor(
            [example["prompt_total_tokens"] for example in examples], dtype=torch.float32
        )
        prompt_group_ids = [example["prompt_group_id"] for example in examples]

        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad(attention_mask, padding_value=0)
        completion_mask = pad(completion_mask, padding_value=0)
        old_log_probs = pad(old_log_probs, padding_value=0)

        # Total valid completion tokens across all samples in the full optimizer-step batch.
        # Values are repeated per sample so slicing keeps the step-level denominator available.
        global_n_tokens = completion_mask.sum()
        global_n_tokens_repeated = torch.full((len(examples),), global_n_tokens.item(), dtype=torch.float32)

        global_num_prompts = len(set(prompt_group_ids))
        global_num_prompts_repeated = torch.full((len(examples),), global_num_prompts, dtype=torch.float32)

        metrics_list = [example["metrics"] for example in examples]
        metrics = (
            {
                key: torch.tensor([m.get(key, 0.0) for m in metrics_list], dtype=torch.float32)
                for key in metrics_list[0]
            }
            if metrics_list and metrics_list[0]
            else {}
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "batch_adv_std": batch_adv_std,
            "prompt_total_tokens": prompt_total_tokens,
            "global_n_tokens": global_n_tokens_repeated,
            "global_num_prompts": global_num_prompts_repeated,
            "metrics": metrics,
        }


class _InfiniteDummyDataset(IterableDataset):
    """Placeholder dataloader source; rollout samples bypass DataLoader entirely."""

    def __iter__(self):
        while True:
            yield {}


class RolloutBatcher:
    """Consumes exactly one optimizer-step batch of online rollout samples."""

    def __init__(
        self,
        *,
        rollout_queue: queue.Queue | None,
        rollout_worker: Any | None,
        accelerator: Any,
        pad_token_id: int,
        per_device_train_batch_size: int,
        max_staleness: int,
        timeout: float,
        advantage_normalization: AdvantageNormalization,
        model_version_fn: Callable[[], int],
    ):
        self.rollout_queue = rollout_queue
        self.rollout_worker = rollout_worker
        self.accelerator = accelerator
        self.pad_token_id = pad_token_id
        self.per_device_train_batch_size = per_device_train_batch_size
        self.max_staleness = max_staleness
        self.timeout = timeout
        self.advantage_normalization = advantage_normalization
        self.model_version_fn = model_version_fn
        self.collator = DataCollatorForRollout(pad_token_id)
        logger.info("inited the RolloutBatcher")

    def get_batch_samples(
        self,
        num_batches: int,
        device: torch.device,
    ) -> tuple[list[dict[str, Any]], torch.Tensor | int | None]:
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        per_device_bs = self.per_device_train_batch_size
        global_microbatch_size = per_device_bs * world_size
        full_batch_size = global_microbatch_size * num_batches

        if self.accelerator.is_main_process:
            examples = self._pull_rollout_examples_for_optimizer_step(full_batch_size)
            full_batch = self._collate_rollout_examples(examples)
        else:
            full_batch = None

        full_batch = self._broadcast_rollout_batch(full_batch, device)

        logger.info(
            "Rollout optimizer-step batch ready: samples=%d seq_len=%d",
            full_batch_size,
            full_batch["input_ids"].shape[1],
        )

        batch_samples = []
        for microbatch_idx in range(num_batches):
            micro_start = microbatch_idx * global_microbatch_size
            micro_end = micro_start + global_microbatch_size
            global_microbatch = self._slice_rollout_batch(full_batch, micro_start, micro_end)

            local_start = rank * per_device_bs
            local_end = local_start + per_device_bs
            local_microbatch = self._slice_rollout_batch(global_microbatch, local_start, local_end)

            batch_samples.append(local_microbatch)

        logger.info(
            "Prepared %d local microbatches: local_batch_size=%d",
            len(batch_samples),
            per_device_bs,
        )

        return batch_samples, None

    def _pull_rollout_examples_for_optimizer_step(
        self,
        full_batch_size: int,
    ) -> list[dict[str, Any]]:
        if self.rollout_queue is None:
            raise RuntimeError("rollout_queue is required on the main process")

        buffer = []
        wait_times = []
        staleness_vals = []

        logger.info(
            "Building rollout optimizer-step batch: full_batch_size=%d per_device_bs=%d "
            "world_size=%d model_version=%d",
            full_batch_size,
            self.per_device_train_batch_size,
            self.accelerator.num_processes,
            self.model_version_fn(),
        )

        while len(buffer) < full_batch_size:
            current_version = self.model_version_fn()
            if self.rollout_worker is not None and hasattr(
                self.rollout_worker,
                "wait_until_next_sample_allowed",
            ):
                self.rollout_worker.wait_until_next_sample_allowed(
                    current_version=current_version,
                    max_staleness=self.max_staleness,
                    timeout=self.timeout,
                )

            if not buffer and self.rollout_queue.qsize() == 0:
                logger.info("queue empty, waiting for rollout samples...")

            t0 = time.time()
            try:
                sample = self.rollout_queue.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise RuntimeError(
                    "Timed out waiting for rollout samples while building an optimizer-step batch"
                ) from exc

            wait_s = time.time() - t0
            current_version = self.model_version_fn()
            staleness = current_version - sample.model_version
            if staleness > self.max_staleness:
                logger.error(
                    "Backpressure invariant failed: rollout sample exceeded max_staleness "
                    f"(staleness={staleness}, max={self.max_staleness}, "
                    f"sample_version={sample.model_version}, current_version={current_version})."
                )

            buffer.append(sample)
            wait_times.append(wait_s)
            staleness_vals.append(staleness)

            logger.info(
                "Rollout optimizer-step buffer size: %d / %d",
                len(buffer),
                full_batch_size,
            )

        if self.advantage_normalization == AdvantageNormalization.BATCH:
            advs = np.array([sample.advantage for sample in buffer], dtype=np.float64)
            batch_adv_std = float(advs.std()) + 1e-8
        else:
            batch_adv_std = 0.0

        return [
            self._rollout_sample_to_dict(sample, wait_s, batch_adv_std, staleness)
            for sample, wait_s, staleness in zip(buffer, wait_times, staleness_vals, strict=True)
        ]

    def _rollout_sample_to_dict(
        self,
        sample: Any,
        queue_wait_time_s: float,
        batch_adv_std: float,
        staleness: int,
    ) -> dict[str, Any]:
        return {
            "input_ids": sample.input_ids,
            "completion_mask": sample.completion_mask,
            "old_log_probs": sample.old_log_probs,
            "advantage": sample.advantage,
            "prompt_total_tokens": sample.prompt_total_tokens,
            "prompt_group_id": sample.prompt_group_id,
            "batch_adv_std": batch_adv_std,
            "metrics": {**sample.metrics, "queue_wait_time_s": queue_wait_time_s, "staleness": float(staleness)},
        }

    def _collate_rollout_examples(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return self.collator(examples)

    def _broadcast_rollout_batch(
        self,
        batch: dict[str, Any] | None,
        device: torch.device,
    ) -> dict[str, Any]:
        if self.accelerator.is_main_process:
            batch_info = [get_data_structure(batch), False]
        else:
            batch_info = [None, False]

        broadcast_object_list(batch_info)

        if not self.accelerator.is_main_process:
            batch = initialize_tensors(batch_info[0])

        batch = send_to_device(batch, device)
        batch = broadcast(batch, from_process=0)
        return batch

    def _slice_rollout_batch(
        self,
        batch: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        return slice_tensors(batch, slice(start, end))
