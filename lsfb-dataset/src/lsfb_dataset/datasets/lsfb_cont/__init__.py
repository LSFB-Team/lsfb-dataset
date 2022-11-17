"""
## lsfb_cont

This module contains classes helping to manipulate the lsfb_cont dataset. The provided classes are :

- **LSFBContConfig** : [dataclass](https://docs.python.org/3/library/dataclasses.html) allowing to configure how the dataset will be loaded
- **LSFBContLandmarks** : Iterator class loading the lsfb_cont landmarks according to the configuration provided.
- **LSFBContLandmarksGenerator** : Generator class loading the lsfb_cont landmarks according to the configuration provided.

"""

from .landmarks import LSFBContLandmarks
from .landmarks_generator import LSFBContLandmarksGenerator
from .config import LSFBContConfig
