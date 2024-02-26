import os
import re

import pandas as pd

from buzzcode.audio import snip_audio


def snip_training(path_annotations, dir_training, dir_raw, cpus, conflict_out='skip'):
    path_metadata = os.path.join(dir_training, "metadata", "metadata_raw.csv")
    dir_audio = os.path.join(dir_training, "audio")

    if os.path.exists(path_metadata):
        quit(f"metadata file already exists at {path_metadata}")

    os.makedirs(os.path.dirname(path_metadata), exist_ok=True)

    annotations = pd.read_csv(path_annotations)

    # kill leading slashes in ident if present; messes up os.path.join
    annotations['ident'] = [re.sub('^/', '', ident) for ident in annotations['ident']]
    annotations['path_raw'] = [os.path.join(dir_raw, ident + '.mp3') for ident in annotations['ident']]

    annotations['path_relative'] = [ident + '_s' + str(start.__floor__()) + '.wav' for ident, start in zip(annotations['ident'], annotations['start'])]
    annotations['duration'] = annotations['end'] - annotations['start']
    annotations['path_snip'] = [os.path.join(dir_audio, ident + "_s" + str(start.__floor__()) + ".wav") for ident, start in zip(annotations['ident'], annotations['start'])]

    annotations.to_csv(path_metadata, index=False)
    sniplist = list(zip(annotations['path_raw'], annotations['path_snip'], annotations['start'], annotations['end']))

    snip_audio(
        sniplist=sniplist,
        cpus=cpus,
        conflict_out='skip'
    )


if __name__ == '__main__':
    snip_training(
        path_annotations='./localData/annotations/2024-02-05/annotations.csv',
        dir_training='./training',
        dir_raw='./localData/raw experiment audio',
        cpus=8,
        conflict_out='skip'
    )
