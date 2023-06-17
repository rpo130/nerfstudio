# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import *
from nerfstudio.data.dataparsers.instant_ngp_dataparser import *
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class LocalNerfDataParserConfig(InstantNGPDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: LocalNerfData)
    """target class to instantiate"""


@dataclass
class LocalNerfData(InstantNGP):
    """Nerfstudio DatasetParser"""

    config: LocalNerfDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        super_dataparser_outputs : DataparserOutputs = super()._generate_dataparser_outputs(split)

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        depth_filenames = []

        for frame in meta["frames"]:
            if "depth_file_path" in frame:
                depth_filepath = PurePath(frame["depth_file_path"])
                depth_fname = data_dir / depth_filepath
                depth_filenames.append(depth_fname)

        dataparser_outputs = DataparserOutputs(
            image_filenames=super_dataparser_outputs.image_filenames,
            cameras=super_dataparser_outputs.cameras,
            scene_box=super_dataparser_outputs.scene_box,
            mask_filenames=super_dataparser_outputs.mask_filenames,
            dataparser_scale=super_dataparser_outputs.dataparser_scale,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": 1,
            },
        )

        return dataparser_outputs