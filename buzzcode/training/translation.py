import numpy as np


def build_translation_dict(translation):
    """
    where key is "from" and value is "to"

    """
    if "from" not in translation.columns or "to" not in translation.columns:
        raise ValueError("DataFrame must contain 'from' and 'to' columns.")

    translation_dict = translation.set_index("from")["to"].to_dict()
    return translation_dict


def translate_labels(labels_raw, translation_dict):
    """
    Translates a list of raw labels using a translation dictionary as built by build_translation_dict.
    Leaves labels unchanged if no key is found (not in translate.csv).
    Drops labels where the translation value is NaN (empty "to" value in translate.csv).

    Args:
        labels_raw (list): The raw labels to translate.
        translation_dict (dict): A dictionary mapping raw labels to their translations.

    Returns:
        list: Translated labels with NaN values removed.
    """
    labels_translated = [
        translation_dict.get(l, l)  # Translate if found, else leave unchanged
        for l in labels_raw
    ]
    # Filter out NaN values
    labels_translated = [label for label in labels_translated if label is not np.nan]
    return labels_translated


def add_labels_translate(s, translation_dict):
    s_up = s.copy()
    s_up.update({'labels_translate': translate_labels(s['labels_raw'], translation_dict)})
    return s_up
