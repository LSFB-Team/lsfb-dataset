import pandas as pd


def get_annotations_in_time_range(df_annotations: pd.DataFrame, time_range: (int, int)):
    return df_annotations[
        ((df_annotations['start'] >= time_range[0]) & (df_annotations['start'] <= time_range[1])) |
        ((df_annotations['end'] >= time_range[0]) & (df_annotations['end'] <= time_range[1]))
    ]
