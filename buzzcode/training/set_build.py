import json
import os

from buzzcode import config as cfg
from buzzcode.embedding.load_embedder import load_embedder
from buzzcode.training.set_extract import extract_set

from buzzcode.training.augment_noise_volume import augment_set
from buzzcode.training.augment_combine import augment_combine_set

from buzzcode.training.set_embed import embed_set


def build_set(setname, embeddername, framehop_prop, event_overlap_prop=0.2, cpus_extract=1, cpus_augment=1, cpus_embed=1, overwrite=False):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    os.makedirs(dir_set, exist_ok=True)

    config_set = {
        'framehop_prop': framehop_prop,
        'event_overlap_prop': event_overlap_prop,
        'embeddername': embeddername
    }

    with open(os.path.join(dir_set, 'config_set.json'), 'x') as f:
        f.write(json.dumps(config_set))


    # ---- build ----
    print('---- starting extraction ----')
    extract_set(setname=setname, cpus=cpus_extract)
    print('---- extraction complete ----')
    
    # ---- augment ----
    # augment_noise_set(setname=setname, cpus=cpus_augment, overwrite=overwrite)
    # augment_volume_set(setname=setname, cpus=cpus_augment, overwrite=overwrite)
    # augment_combine_set(setname=setname, cpus=cpus_augment, limit=2, prop_combine=0.8)

    print('---- starting embedding ----')
    embed_set(setname=setname, cpus=cpus_embed, overwrite=overwrite)
    print('---- embedding complete ----')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    build_set(
        setname='standard',
        embeddername='yamnet',
        framehop_prop=0.96*0.1,
        event_overlap_prop=0.2,
        cpus_extract=1,
        cpus_embed=1,
        overwrite=False
    )