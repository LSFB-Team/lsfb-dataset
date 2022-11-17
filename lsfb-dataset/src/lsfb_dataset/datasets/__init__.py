"""
# Module Datasets

This module provides configurations classes and iterators for the lsfb_iso and lsfb_cont dataset.
The module is organised in two submodules : One for the continuous dataset and one for the isolated one.

"""

from .lsfb_cont import LSFBContConfig, LSFBContLandmarks, LSFBContLandmarksGenerator
from .lsfb_isol import LSFBIsolConfig, LSFBIsolLandmarks, LSFBIsolLandmarksGenerator
