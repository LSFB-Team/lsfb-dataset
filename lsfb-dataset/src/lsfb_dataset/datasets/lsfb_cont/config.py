from dataclasses import dataclass
from typing import Optional, Union, Sequence


@dataclass
class LSFBContConfig:
    """
        Simple configuration class for the LSFB_CONT dataset.
        Each instance of the LSFB_CONT dataset is a video with corresponding annotations.
        Each annotation is made up of a segment (start, end) and a label.

        LSFB: French Belgian Sign Language
        CONT: Continuous videos in LSFB

        See `LSFBContLandmarks` to load the dataset.

        Args:
            root: Root directory of the LSFB_CONT dataset.
                The dataset must already be downloaded !
            landmarks: Select which landmarks (features) to use.
                'face' for face mesh (468 landmarks);
                'pose' for pose skeleton (23 landmarks);
                'left_hand' for left hand skeleton (21 landmarks);
                'right_hand' for right hand skeleton (21 landmarks);
                Default = ('pose', 'left_hand', 'right_hand').
            use_3d: If true, use 3D landmarks. Otherwise, use 2D landmarks.
                Default=False.
            use_raw: If true, use raw landmarks. Otherwise, use preprocessed landmarks where:
                - missing landmarks are interpolated;
                - vibrations have been reduced by using smoothing (Savitchy Golay filter).
                Default=False.

            features_transform: Callable object used to transform the features.
            target_transform: Callable object used to transform the targets.
            transform: Callable object used to transform both the features and the targets.

            split: Specify which subset of the dataset is used.
                'fold_x' where x is in {0, 1, 2, 3, 4} for a specific fold;
                'train' for training set (folds 2, 3, 4);
                'test' for the test set (folds 0 and 1);
                'all' for all the instances of the dataset (all folds);
                'mini_sample' for a tiny set of instances (10 instances).
                Default = 'all'.

            hands: Only load the sign of a specific hand, or both of them.
                'right' for the signs from the right hand of the signer;
                'left' for the signs from the left hand of the signer;
                'both' for the signs from both hands of the signer.
                Default = 'both'.
            segment_level: Specifies the level at which annotations are extracted.
                'signs'
                'subtitles'
                Default = 'signs'
            segment_label: Specify which label to use for each segment.
                'sign_gloss': the gloss of the signs is used. Example: HAND
                'sign_index': the index (class) of the signs is used. Example: 45
                'text': the text of the sign, or subtitle, is used. This can be useful when annotations are subtitles.
                If the annotations are subtitles, multiple labels are assigned to a segment.
                For example, a sequence of glosses, indices or event the full text of the subtitles.
                Default='sign_index'
            segment_unit: Specify which unit is used for the boundaries (start, end) of the annotations.
                'frame': frame indices in annotations boundaries.
                'ms': milliseconds in annotations boundaries.
                default='ms'

            n_labels: If this parameter is an integer `x`, then `x+1` labels are used for the `x` most
                frequent signs and the background label. If none, the number of labels is the number of different signs
                in the dataset no matter their occurrences.
                This parameter is used to filter out signs with very few examples.
                Default=750

            window: Optional argument (window_size, window_stride) to use fixed-size windows instead of
                variable length sequences.
                If specified, the dataset is windowed with a window size and a window stride.
                Be careful, this argument changes the number of instances in the dataset !
                Default = None.

            show_progress: If true, shows a progress bar while the dataset is loading. Default = True.

        Author:
            ppoitier (v 2.0)
    """

    root: str
    landmarks: Optional[tuple[str, ...]] = ('pose', 'left_hand', 'right_hand')
    use_3d: bool = False
    use_raw: bool = False

    features_transform: callable = None
    target_transform: callable = None
    transform: callable = None

    split: str = 'all'

    hands: str = 'both'
    segment_level: str = 'signs'
    segment_label: str = 'sign_index'
    segment_unit: str = 'ms'

    n_labels: Union[int, None] = 750

    window: Optional[tuple[int, int]] = None

    show_progress: bool = True
