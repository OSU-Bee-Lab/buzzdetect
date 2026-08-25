import numpy as np
import pandas as pd


def add_time(df, time_start, framehop_s, digits_time):
    df.insert(
        column='start',
        value=range(len(df)),
        loc=0
    )

    df['start'] = df['start'] * framehop_s
    if time_start != 0:
        df['start'] = df['start'] + time_start

    df['start'] = round(df['start'], digits_time)
    return df


def format_detections(results, threshold, classes, framehop_s, digits_time, time_start):
    buzz_index = classes.index('ins_buzz')
    activations = results[:, buzz_index]
    detections = activations > threshold
    detections = detections.astype(int)

    df = pd.DataFrame(detections, columns=['detections_ins_buzz'])
    df = add_time(df, time_start, framehop_s, digits_time)

    return df


def format_activations(results, classes, framehop_s, digits_time, time_start=0, classes_keep='all', digits_results=2):
    results = np.array(results).round(digits_results)

    classes_out = classes.copy()

    if classes_keep != 'all':
        classes_unknown = set(classes_keep) - set(classes)
        if classes_unknown:
            raise ValueError(f"Bad classes in classes_keep: {', '.join(list(classes_unknown))}")

        keep_indices = [i for i, cls in enumerate(classes) if cls in classes_keep]
        results = results[:, keep_indices]
        classes_out = [classes[i] for i in keep_indices]

    df = pd.DataFrame(results)
    df.columns = ['activation_' + c for c in classes_out]
    df = add_time(df, time_start, framehop_s, digits_time)

    return df