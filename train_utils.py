import pickle
import os

import numpy as np
import bz2
from utils_load import load_file


def sample_weights_and_metrics(evolved_weights, run_metrics, sample_epochs):

    n_evolved_weights = len(evolved_weights)
    n_sample_epochs = len(sample_epochs)
    sample_weights = []
    sample_metrics = np.zeros((n_sample_epochs, 4))

    for i, epoch in enumerate(sample_epochs):
        if i < n_evolved_weights:
            current_weights = evolved_weights[i]
            sample_weights.append(current_weights)
            sample_metrics[i] = run_metrics[i]
        else:
            break

    return sample_weights, sample_metrics


def load_set_trained(fname, sample_epochs, use_all=True) -> dict:
    if os.path.isfile(fname):
        try:
            set_pretrained_samples = {}
            set_pretrained = load_file(fname)

            if use_all:
                return set_pretrained

            set_pretrained_samples['density_levels'] = set_pretrained['density_levels']
            set_pretrained_samples['runs'] = []

            for run in set_pretrained['runs']:
                old_run = run['run']
                new_run = {'run_id': old_run['run_id'], 'set_params': old_run['set_params'],
                           'training_time': old_run['training_time']}

                evolved_weights = old_run['evolved_weights']
                run_metrics = old_run['set_metrics']

                new_run['evolved_weights'], new_run['set_metrics'] = sample_weights_and_metrics(evolved_weights, run_metrics, sample_epochs)

                set_pretrained_samples['runs'].append({'set_sparsity': run['set_sparsity'], 'run': new_run})

            del set_pretrained
            return set_pretrained_samples

        except EOFError:
            print(f"FILE malformed: {fname} ")
            return {}
    else:
        print(f"FILE: {fname} already processed or non-existent -> skipping")
        return {}
