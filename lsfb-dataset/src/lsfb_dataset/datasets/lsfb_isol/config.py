from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LSFBIsolConfig:
    """
    Simple configuration class for the LSFB ISOL Dataset.

    LSFB: French Belgian Sign Language
    ISOL: isolated videos in LSFB

    Args:
        root: Root directory of the LSFB_ISOL dataset.
            The dataset must already be downloaded!
        landmarks: Select which landmarks (features) to use.
            'face' for face mesh (468 landmarks);
            'pose' for pose skeleton (23 landmarks);
            'left_hand' for left hand skeleton (21 landmarks);
            'right_hand' for right hand skeleton (21 landmarks);
            Default=['pose', 'left_hand', 'right_hand'].
        use_3d: If true, use 3D landmarks. Otherwise, use 2D landmarks.
            Default=False.
        use_raw: If true, use raw landmarks. Otherwise, use preprocessed landmarks where:
            - missing landmarks are interpolated;
            - vibrations have been reduced by using smoothing (Savitchy Golay filter).
            Default=False.

        target: TODO COMPLETE
            'sign_gloss': ...;
            'sign_index': ...;
            Default='sign_index'.

        transform: Callable object used to transform the features.

        split: Specify which subset of the dataset is used.
            'fold_x' where x is in {0, 1, 2, 3, 4} for a specific fold;
            'train' for training set (folds 2, 3, 4);
            'test' for the test set (folds 0 and 1);
            'all' for all the instances of the dataset (all folds);
            'mini_sample' for a tiny set of instances (10 instances).
            Default='all'.

        sequence_max_length: (Optional) Max length of the clip sequence. Default=50.

        n_labels: If this parameter is an integer `x`, then `x+1` labels are used for the `x` most
            frequent signs and the background label. If none, the number of labels is the number of different signs
            in the dataset no matter their occurrences.
            This parameter is used to filter out signs with very few examples.
            Default=750.

        show_progress: If true, shows a progress bar while the dataset is loading. Default = True.


    Author:
        jfink (v 1.0)
        ppoitier (v 2.0)
    """

    root: str
    landmarks: Optional[list[str]] = ("pose", "left_hand", "right_hand")
    use_3d: bool = False
    use_raw: bool = False

    target: str = "sign_index"

    transform: callable = None

    split: str = "all"
    n_labels: int = 750
    sequence_max_length: int = 50

    show_progress: bool = True
