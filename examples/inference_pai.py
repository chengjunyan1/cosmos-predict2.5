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
import sys,os
import tempfile
import json

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


class PAIGEvalMaker:
    def __init__(self):
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))
        self.cache_dir = os.path.join(hf_home, "pai-bench", "generation")

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
    
    def get_cosmos_predict_json_file(self, index, seed=0):
        json_data = self.get_cosmos_predict_json(index, seed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json.dumps(json_data).encode("utf-8"))
            return Path(temp_file.name)

    def get_cosmos_predict_json_files(self, indices, seed=0):
        json_files = []
        for index in indices:
            json_files.append(self.get_cosmos_predict_json_file(index, seed))
        return json_files



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
    paig_maker = PAIGEvalMaker()
    if args.overrides.seed is not None:
        seed = args.overrides.seed
    else:
        seed = 0
    input_files = paig_maker.get_cosmos_predict_json_files(args.input_indices, seed)
    inference_samples = InferenceArguments.from_files(input_files, overrides=args.overrides, setup_args=args.setup)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.inference import Inference

    inference = Inference(args.setup)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    """
    see https://github.com/chengjunyan1/cosmos-predict2.5/blob/main/docs/inference.md, major difference is -i option is now pai bench indices

    Example usage:
    python examples/inference_pai.py -i 123 -o outputs/pai_eval_output/test 
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
