"""
## lsfb_isol

This module contains classes helping to manipulate the lsfb_isol dataset. The provided classes are :

- **LSFBIsolConfig** : [dataclass](https://docs.python.org/3/library/dataclasses.html) allowing to configure
how the dataset will be loaded
- **LSFBIsolLandmarks** : Iterator class loading the lsfb_isol landmarks according to the configuration provided.
- **LSFBIsolLandmarksGenerator** : Generator class loading the lsfb_isol landmarks according to the configuration
provided.

"""

from .landmarks import LSFBIsolLandmarks
from .landmarks_generator import LSFBIsolLandmarksGenerator
from .config import LSFBIsolConfig
