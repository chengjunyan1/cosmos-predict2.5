# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base model inference script."""

import sys
import json
import os
import tempfile
import time
import torch
from pathlib import Path
from typing import Annotated

from cosmos_oss.init import init_output_dir

from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    is_rank0,
)



def inference(
    input_files: list[Path],
    setup: SetupArguments,
    overrides: InferenceOverrides
):
    # inference with performance metrics
    inference_samples = InferenceArguments.from_files(input_files, overrides=overrides, setup_args=setup)
    init_output_dir(setup.output_dir, profile=setup.profile)

    from cosmos_predict2.inference import Inference

    inference = Inference(setup)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    inference.generate(inference_samples, output_dir=setup.output_dir)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Only Rank 0 processes and prints the metrics to avoid terminal spam
    if is_rank0():
        total_time = end_time - start_time
        num_samples = len(inference_samples)
        avg_time = total_time / num_samples if num_samples > 0 else 0
        
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_vram_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Print to console
        print("\n" + "="*40)
        print("🚀 INFERENCE BENCHMARK RESULTS")
        print("="*40)
        print(f"Total Time ({num_samples} videos) : {total_time:.2f} seconds")
        print(f"Average Time per Video  : {avg_time:.2f} seconds")
        print(f"Peak VRAM Allocated     : {peak_vram_gb:.2f} GB")
        print(f"Total VRAM Reserved     : {reserved_vram_gb:.2f} GB")
        print("="*40 + "\n")

        json_path = setup.output_dir / "benchmark_metrics.json"
        
        with open(json_path, mode='w') as f:
            json.dump({
                "model": setup.model,
                "world_size": os.environ.get("WORLD_SIZE", "1"),
                "num_samples": num_samples,
                "total_time": total_time,
                "avg_time": avg_time,
                "peak_vram_gb": peak_vram_gb,
                "reserved_vram_gb": reserved_vram_gb,
                "input_files": input_files
            }, f, indent=4)


