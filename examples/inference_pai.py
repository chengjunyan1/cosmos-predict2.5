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

# Stub wandb before any cosmos_predict2 imports trigger its import chain.
# The chain inference.py → guardrail → imaginaire/config.py → trainer.py
# → callback.py does `import wandb` unconditionally, costing ~20s at startup.
# Wandb is not used during inference.
if "wandb" not in sys.modules:
    import types as _types

    class _WandbStub(_types.ModuleType):
        def __getattr__(self, name: str):
            return _WandbStub(f"wandb.{name}")
        def __call__(self, *args, **kwargs):
            return None
        def __bool__(self):
            return True

    sys.modules["wandb"] = _WandbStub("wandb")
    del _types, _WandbStub

import json
import os
import tempfile
import time
import torch
from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
import torch.distributed as dist

from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)

from datasets import load_dataset
from huggingface_hub import snapshot_download

# setup from PAI benchmark paper
# Model                 Version             #Frames     FPS     Width   Height
# Cosmos-Predict2.5-2B  base/post-trained   93          16      1280    704

# output_dir required for cosmos-predict2.5
# /path/to/your/videos/ (the output path, -o option)
# ├── {video_id}__{seed}.mp4
# ├── {video_id}__{seed}.mp4
# └── ... (more video files following the naming pattern)

# example of a row in the PAI benchmark dataset
# {'prompt_en': 'A video capturing a heavy rainfall in a tropical or subtropical environment. The scene opens with a close-up of a tree with large, broad leaves that are visibly wet and glistening under the rain. The leaves are attached to dark, slender branches that sway vigorously in the wind, indicating the strength of the storm. In the background, other trees with thinner trunks and more sparse foliage are partially obscured by the mist created by the rain, adding depth to the scene. The sky is overcast, contributing to the gray and damp atmosphere. As the video progresses, the rain continues to pour, with water droplets continuously falling and splashing against the leaves and branches. The leaves rustle and move more vigorously, and the branches bend further under the force of the wind. No new objects appear throughout the video; the focus remains on the interaction between the rain, wind, and vegetation. By the final frame, the scene retains its initial characteristics: the wet, swaying leaves, the misty background, and the persistent, heavy rain, maintaining the overall mood of a stormy day.',
#  'dimension': ['aesthetic_quality',
#   'background_consistency',
#   'imaging_quality',
#   'motion_smoothness',
#   'overall_consistency',
#   'subject_consistency',
#   'i2v_background',
#   'i2v_subject'],
#  'image_name': 'misc_933704-uhd_3840_2160_30fps.jpg',
#  'reference_video': 'misc_933704-uhd_3840_2160_30fps.mp4',
#  'video_id': 'misc_933704-uhd_3840_2160_30fps'}


def _fast_hf_download(cmd_args: list[str]) -> str:
    """Drop-in replacement for checkpoint_db._hf_download using the Python API.

    The original shells out to `uvx hf download ...` twice per checkpoint — once
    to fetch/verify and once with --quiet to retrieve the local path. The uvx
    process startup + HF CLI verification adds 5-7s per call even on cache hits.
    The Python huggingface_hub API resolves cached paths in milliseconds.

    cmd_args is the argument list built by CheckpointFileHf / CheckpointDirHf:
        [repo_id, "--repo-type", "model", "--revision", rev,
         filename | --include pat... [--exclude pat...]]
    """
    from huggingface_hub import hf_hub_download, snapshot_download as hf_snapshot_download

    repo_id = cmd_args[0]
    revision: str | None = None
    include_patterns: list[str] = []
    exclude_patterns: list[str] = []
    filename: str | None = None

    i = 1
    while i < len(cmd_args):
        tok = cmd_args[i]
        if tok == "--repo-type":
            i += 2
        elif tok == "--revision":
            revision = cmd_args[i + 1]
            i += 2
        elif tok == "--include":
            i += 1
            while i < len(cmd_args) and not cmd_args[i].startswith("--"):
                include_patterns.append(cmd_args[i])
                i += 1
        elif tok == "--exclude":
            i += 1
            while i < len(cmd_args) and not cmd_args[i].startswith("--"):
                exclude_patterns.append(cmd_args[i])
                i += 1
        elif not tok.startswith("--"):
            filename = tok
            i += 1
        else:
            i += 1

    if filename is not None:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="model",
        )
    else:
        return hf_snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
            allow_patterns=include_patterns or None,
            ignore_patterns=exclude_patterns or None,
        )


class PAIGEvalMaker:
    def __init__(self):
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))
        self.cache_dir = os.path.join(hf_home, "pai-bench", "generation")
        self._temp_files: list[str] = []

        # Check if we are running in a multi-GPU torchrun environment
        is_dist = dist.is_available() and dist.is_initialized()

        # If distributed, force Ranks 1-7 to pause and wait here
        if is_dist and dist.get_rank() != 0:
            dist.barrier()

        # Only Rank 0 (or a single-GPU run) performs the network downloads
        if not is_dist or dist.get_rank() == 0:
            snapshot_download(
                repo_id="shi-labs/physical-ai-bench-generation",
                repo_type="dataset",
                allow_patterns="condition_image/*",
                local_dir=self.cache_dir
            )
            self.ds = load_dataset("shi-labs/physical-ai-bench-generation")

        # Rank 0 reaches this point and unlocks the barrier for Ranks 1-7
        if is_dist and dist.get_rank() == 0:
            dist.barrier()

        # Ranks 1-7 resume and instantly load from the local disk cache
        if is_dist and dist.get_rank() != 0:
            self.ds = load_dataset("shi-labs/physical-ai-bench-generation")

    @property
    def length(self):
        return len(self.ds['benchmark'])

    @property
    def image_dir(self):
        return os.path.join(self.cache_dir, "condition_image")

    def get_cosmos_predict_json(self, index, seed=0):
        row = self.ds['benchmark'][index]
        video_id = row['video_id']
        image_name = row['image_name']
        image_path = os.path.join(self.image_dir, image_name)
        prompt = row['prompt_en']

        json_data = {
            "inference_type": "image2world",
            "name": f"{video_id}__{seed}",
            "seed": seed,
            "num_output_frames": 93,
            "resolution": "1280,704",
            "input_path": image_path,
            "prompt": prompt,
        }
        return json_data

    def get_cosmos_predict_json_file(self, index, seed=0) -> Path:
        json_data = self.get_cosmos_predict_json(index, seed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            f.write(json.dumps(json_data).encode("utf-8"))
            self._temp_files.append(f.name)
            return Path(f.name)

    def get_cosmos_predict_json_files(self, indices, seed=0) -> list[Path]:
        """Create temp JSON input files for the given indices.

        Only rank 0 creates the files; other ranks receive the paths via broadcast
        so all ranks share the same files without redundant I/O.
        """
        is_dist = dist.is_available() and dist.is_initialized()
        paths: list[str | None] = [None] * len(indices)

        if not is_dist or dist.get_rank() == 0:
            for i, index in enumerate(indices):
                paths[i] = str(self.get_cosmos_predict_json_file(index, seed))

        if is_dist:
            dist.broadcast_object_list(paths, src=0)

        return [Path(p) for p in paths]  # type: ignore[arg-type]

    def cleanup(self) -> None:
        """Delete all temp files created by this instance."""
        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    # input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    # """Path to the inference parameter file(s).
    # If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    # """
    input_indices: Annotated[list[int], tyro.conf.arg(aliases=("-i",))]
    """Indices of the PAI benchmark samples to run. example: -i 0 1 2 (separated by space)
    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""


def main(
    args: Args,
):
    # Replace the uvx-subprocess HF downloader with the Python API before
    # Inference.__init__ triggers it. Each uvx call costs 5-7s in process
    # startup + HF CLI network verification even on cache hits.
    from cosmos_predict2._src.imaginaire.utils import checkpoint_db
    checkpoint_db._hf_download = _fast_hf_download

    paig_maker = PAIGEvalMaker()
    seed = args.overrides.seed if args.overrides.seed is not None else 0
    input_files = paig_maker.get_cosmos_predict_json_files(args.input_indices, seed)

    try:
        inference_samples = InferenceArguments.from_files(input_files, overrides=args.overrides, setup_args=args.setup)
        init_output_dir(args.setup.output_dir, profile=args.setup.profile)

        from cosmos_predict2.inference import Inference

        inference = Inference(args.setup)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        inference.generate(inference_samples, output_dir=args.setup.output_dir)

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

            json_path = args.setup.output_dir / "benchmark_metrics.json"
            
            with open(json_path, mode='w') as f:
                json.dump({
                    "model": args.setup.model,
                    "world_size": os.environ.get("WORLD_SIZE", "1"),
                    "num_samples": num_samples,
                    "total_time": total_time,
                    "avg_time": avg_time,
                    "peak_vram_gb": peak_vram_gb,
                    "reserved_vram_gb": reserved_vram_gb,
                    "input_indices": args.input_indices
                }, f, indent=4)
    finally:
        paig_maker.cleanup()


if __name__ == "__main__":
    """
    see https://github.com/chengjunyan1/cosmos-predict2.5/blob/main/docs/inference.md, major difference is -i option is now pai bench indices

    Example usage:
    python examples/inference_pai.py -i 0 -o outputs/pai_eval_output/test
    python examples/inference_pai.py -i 0 1 2 -o outputs/pai_eval_output/test
    python examples/inference_pai.py -i 0 1 2 -o outputs/pai_eval_output/test --model 2B/post-trained --seed 42

    for multi-gpu, use torchrun:
    torchrun --nproc_per_node=8 examples/inference_pai.py -i 0 1 2 -o outputs/pai_eval_output/test --model 2B/post-trained --seed 42
    """
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
