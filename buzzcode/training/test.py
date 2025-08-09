import glob
import os
import pickle
import re
import warnings
import json

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import buzzcode.config as cfg
from buzzcode.analysis.analysis import load_model_config, load_model, format_activations, get_framelength_digits
from buzzcode.embedders import load_embedder_model, load_embedder_config
from buzzcode.utils import search_dir, build_ident

framehop_prop_test = 1

def build_test_embeddings(embeddername):
    embeddername = embeddername.lower()
    embedder_config = load_embedder_config(embeddername)
    embedder = load_embedder_model(embeddername=embeddername, framehop_s = embedder_config['framelength']*framehop_prop_test)

    dir_test = cfg.TRAIN_DIR_TESTS_AUDIO
    paths_audio = search_dir(dir_test, list(sf.available_formats().keys()))

    dir_embeddings = os.path.join(cfg.TRAIN_DIR_TESTS_EMBEDDINGS, embeddername)

    for path_audio in paths_audio:
        print(f'caching test embedding for {path_audio}')
        path_out = re.sub(dir_test, dir_embeddings, path_audio)
        path_out = os.path.splitext(path_out)[0] + '.pickle'

        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        track = sf.SoundFile(path_audio)
        samples = track.read()

        if track.channels > 1:
            samples = np.mean(samples, axis=1)

        samples = librosa.resample(y=samples, orig_sr=track.samplerate, target_sr=embedder_config['samplerate'])
        embeddings = embedder(samples)

        with open(path_out, 'wb') as f:
            pickle.dump(embeddings, f)

def analyze_test_embeddings(modelname):
    config_model = load_model_config(modelname)
    config_embedder = load_embedder_config(config_model['embeddername'])
    dir_embeddings = os.path.join(cfg.TRAIN_DIR_TESTS_EMBEDDINGS, config_model['embeddername'])
    dir_out = os.path.join(cfg.DIR_MODELS, modelname, 'tests', 'results')

    paths_embeddings = search_dir(dir_embeddings, ['.pickle'])
    if not paths_embeddings:
        print(f"TEST: no test embeddings found for embedder {config_model['embeddername']}; skipping test")
        return None

    try:
        model = load_model(modelname)
    except FileNotFoundError:
        warnings.warn(f"TEST: model {modelname} not found; skipping test")
        return None

    test_results = pd.DataFrame()
    for path_embeddings in paths_embeddings:
        ident = build_ident(path=path_embeddings, root_dir=dir_embeddings)

        print(f'TEST: analyzing test embedding for {path_embeddings}')
        with open(path_embeddings, 'rb') as f:
            embeddings = pickle.load(f)

        results = model(embeddings)

        output = format_activations(
            results,
            classes=config_model['classes'],
            framehop_s=config_embedder['framelength']*framehop_prop_test,
            digits_time=get_framelength_digits(config_embedder['framelength']),
            classes_keep=['ins_buzz'],
            digits_results=3
        )

        output.insert(loc=0, column='ident', value=ident)

        test_results = pd.concat([test_results, output])

    return test_results

def analyze_test_audio(modelname):
    """ EXPERIMENTAL, for models that directly analyze audio"""
    config_model = load_model_config(modelname)
    dir_audio = os.path.join(cfg.TRAIN_DIR_TESTS_AUDIO)
    dir_out = os.path.join(cfg.DIR_MODELS, modelname, 'tests', 'results')

    paths_audio = search_dir(dir_audio, ['.mp3'])

    try:
        model = load_model(modelname)
    except FileNotFoundError:
        warnings.warn(f"TEST: model {modelname} not found; skipping test")
        return None

    for path_audio in paths_audio:
        print(f'TEST: analyzing test audio for {path_audio}')
        audio_data, sr_native = sf.read(path_audio)
        audio_data = librosa.resample(y=audio_data, orig_sr=sr_native, target_sr=config_model['samplerate'])

        results = model(audio_data)

        output = format_activations(
            results,
            classes=config_model['classes'],
            framehop_s=config_model['framelength_s']*framehop_prop_test,
            digits_time=get_framelength_digits(config_model['framelength_s']),
            classes_keep=['ins_buzz'],
            digits_results=5
        )

        path_out = re.sub(dir_audio, dir_out, path_audio)
        path_out = os.path.splitext(path_out)[0] + cfg.SUFFIX_RESULT_COMPLETE

        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        output.to_csv(path_out, index=False)

    return None

def build_annotations():
    paths_annotations = glob.glob(cfg.TRAIN_DIR_TESTS_ANNOTATIONS + '/**/*.txt', recursive=True)

    annotations = pd.DataFrame()
    for path_annotations in paths_annotations:
        ident = build_ident(path=path_annotations, root_dir=cfg.TRAIN_DIR_TESTS_ANNOTATIONS)

        annotation = pd.read_csv(path_annotations, sep='\t', header=None)
        annotation.insert(loc=0, column='ident', value=ident)
        
        annotation.columns =  ['ident', 'start', 'end', 'label']

        annotations = pd.concat([annotations, annotation])

    return annotations


def join_results(results, annotations, framelength, overlap_prop):
    """
    Join results and annotations dataframes based on overlapping time windows.

    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame with columns: ident, start, ins_buzz, as returned by analyze_test_audio() or analyze_test_embeddings()
    annotations : pd.DataFrame
        DataFrame with columns: ident, start, end, label, as returned by build_annotations()
    framelength : float
        Duration of a single frame of results, in seconds
    overlap_prop : float
        The minimum proportion of a frame occupied by an event to be considered positive

    Returns:
    --------
    pd.DataFrame
        Merged dataframe with columns: ident, start, ins_buzz, label, correct
    """

    annotations = annotations.copy()
    results = results.copy()

    overlap_s = framelength * overlap_prop

    annotations['start_overlap'] = annotations['start'] + overlap_s
    annotations['end_overlap'] = annotations['end'] - overlap_s

    # Perform the overlap join
    results_join = []
    for _, result_row in results.iterrows():
        # Find overlapping annotations for this result row
        # An overlap exists if:
        # - annotation effective start <= result end, AND
        # - annotation effective end >= result start
        overlapping_annotations = annotations[
            (annotations['start_overlap'] <= result_row['start'] + framelength) &
            (annotations['end_overlap'] >= result_row['start']) &
            (annotations['ident'] == result_row['ident'])
            ]

        if len(overlapping_annotations) > 0:
            # Combine all overlapping labels
            labels = overlapping_annotations['label'].dropna().unique()
            combined_label = '; '.join(sorted(labels)) if len(labels) > 0 else None
        else:
            combined_label = 'ambient_background'

        # Create result row
        merged_row = {
            'ident': result_row['ident'],
            'start_frame': result_row['start'],
            'ins_buzz': result_row['ins_buzz'],
            'label': combined_label
        }

        results_join.append(merged_row)

    results_join = pd.DataFrame(results_join)
    results_join['correct'] = [bool(re.search('ins_buzz', l)) for l in results_join['label']]

    return results_join

def sx(results_join, precision_requested):
    ''' get sensitivity and threshold at x precision'''
    results_join = results_join.sort_values(by='ins_buzz', ascending=False, ignore_index=True)
    results_join['true_positives'] = results_join['correct'].cumsum()
    results_join['precision'] = results_join['true_positives'] / (results_join.index + 1)
    results_join['delta'] = results_join['precision'] - precision_requested

    over = results_join[results_join['delta'] > 0].sort_values(by='delta', ascending=True, inplace=False)
    under = results_join[results_join['delta'] < 0].sort_values(by='delta', ascending=False, inplace=False)

    if len(over) == 0:
        threshold = under.iloc[0]['ins_buzz']
    elif len(under) == 0:
        threshold = over.iloc[0]['ins_buzz']
    else:
        threshold = over.iloc[0]['ins_buzz'] + (under.iloc[0]['ins_buzz'] - over.iloc[0]['ins_buzz']) / 2


    sensitivity = np.mean(results_join[results_join['correct']]['ins_buzz'] > threshold)

    return {'threshold': threshold.__round__(2), 'precision': precision_requested.__round__(2), 'sensitivity': sensitivity.__round__(2)}


def test_model(modelname, overwrite=False):
    dir_test_out = os.path.join(cfg.DIR_MODELS, modelname, 'tests')
    path_sx = os.path.join(dir_test_out, 'sx.csv')

    if os.path.exists(path_sx) and not overwrite:
        print(f'TEST: model "{modelname}" already tested; skipping test')
        return None

    os.makedirs(dir_test_out, exist_ok=True)

    config_model = load_model_config(modelname)
    config_embedder = load_embedder_config(config_model['embeddername'])

    with open(os.path.join(cfg.DIR_MODELS, modelname, 'config_set.txt'), 'r') as f:
        config_set = json.load(f)

    dir_test_embeddings = os.path.join(cfg.TRAIN_DIR_TESTS_EMBEDDINGS, config_model['embeddername'])
    if not os.path.exists(dir_test_embeddings):
        build_test_embeddings(config_model['embeddername'])

    results = analyze_test_embeddings(modelname)
    annotations = build_annotations()

    results_join = join_results(
        results=results,
        annotations=annotations,
        framelength=config_embedder['framelength'],
        overlap_prop=config_set['event_overlap_prop']
    )

    results_join.to_csv(os.path.join(dir_test_out, 'results.csv'), index=False)

    df_sx = pd.DataFrame([sx(results_join, p) for p in [0.9, 0.95, 0.99]])
    df_sx.to_csv(path_sx, index=False)

    return None


def pull_sx(modelname, precision_requested):
    dir_tests = os.path.join(cfg.DIR_MODELS, modelname, 'tests')
    try:
        df_sx = pd.read_csv(os.path.join(dir_tests, 'sx.csv'))

    except FileNotFoundError:
        raise FileNotFoundError(f'tests not available for model "{modelname}"; run test_model({modelname}) and proceed') from None

    if precision_requested in df_sx['precision'].values:
        return df_sx[df_sx['precision'] == precision_requested].iloc[0].to_dict()

    return sx(pd.read_csv(os.path.join(dir_tests, 'results.csv')), precision_requested)
