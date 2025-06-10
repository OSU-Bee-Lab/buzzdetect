from buzzcode.utils import setthreads
setthreads(8)

import json
import os
import shutil

from buzzcode import config as cfg
from buzzcode.embedders import load_embedder_config
from buzzcode.training.augment_combine import combine_set
from buzzcode.training.extract_embeddings import embed_set
from buzzcode.training.extract_samples import extract_set
from buzzcode.training.training import clean_name


def create_set(setname, annotationname, embeddername, framehop_prop, foldname, event_overlap_prop=0.2,
               augmentname_combine=None, cpus_extract=7, cpus_embed=1, combine_limit=10):
    dir_set = os.path.join(cfg.DIR_TRAIN_SET, setname)
    os.makedirs(dir_set)

    # ---- copy files to set directory ----
    # annotations
    annotationname = clean_name(annotationname, '^annotations_', '\\.csv')
    shutil.copy(
        os.path.join(cfg.DIR_TRAIN_ANNOTATION, 'annotations_' + annotationname + '.csv'),
        os.path.join(dir_set, 'annotations.csv')
    )

    # folds
    shutil.copy(
        os.path.join(cfg.DIR_TRAIN_FOLD, 'folds_' + foldname + '.csv'),
        os.path.join(dir_set, 'folds.csv')
    )

    # embedder config
    config_embedder = load_embedder_config(embeddername)
    with open(os.path.join(dir_set, 'config_embedder.txt'), 'x') as f:
        f.write(json.dumps(config_embedder))

    # set config
    framehop_s = framehop_prop * config_embedder['framelength']
    config = {
        'set': setname,
        'annotation': annotationname,
        'embedder': embeddername,
        'event_overlap_prop': event_overlap_prop,
        'framehop_s': framehop_s
    }

    # combination
    if augmentname_combine is not None:
        shutil.copy(
            os.path.join(cfg.DIR_TRAIN_AUGMENT, 'combine_' + augmentname_combine + '.csv'),
            os.path.join(dir_set, 'augmentation_combine.csv')
        )

        config.update({'augmentation_combine': augmentname_combine})

    with open(os.path.join(dir_set, 'config_set.txt'), 'x') as f:
        f.write(json.dumps(config))

    # ---- build ----
    extract_set(setname=setname, event_overlap_prop=event_overlap_prop, framehop_s=framehop_s, cpus=cpus_extract)
    print('---- extraction complete ----')

    print('---- starting embedding ----')
    embed_set(setname=setname, cpus=cpus_embed)
    print('---- embedding complete ----')

    print('---- starting combination ----')
    combine_set(setname=setname, limit=combine_limit, cpus=cpus_embed)
    print('---- combination complete ----')

    print('---- set done :D ----')

