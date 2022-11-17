from typing import Literal, NewType


DataSubset = NewType('DataSubset', Literal['all', 'train', 'test', 'mini_sample'])
Hand = NewType('Hand', Literal['left', 'right', 'both'])
Target = NewType('Target', Literal['activity', 'signs', 'signs_and_transitions'])
