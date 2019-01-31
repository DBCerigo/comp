#!/usr/bin/env python3
"""
Module for validation computation and validation results storing+reading
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn import model_selection
from sklearn.metrics import matthews_corrcoef

# set random seed for reproduce
np.random.seed(41732)
# WARN: if using multithreading see https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener

TEST_CUTOFF = 100


def run(model, hyperparams=None):
    """
    Main validation function.

    Parameters
    ----------
    model: obj
        Must follow sklearn `estimator` api.
        (I.e. implement `fit` and `predict` functions.)
    hyperparams: dict
        Optional dict of hyperparameters

    Returns
    --------
    Validation results object.

    Writes
    -------
    Validation results to `/results`
    """
    scores = cross_val_score(
            model,
            read_data(),
            read_targets(),
            groups=read_metadata().id_measurement,
            scoring=matthews_corrcoef,
            cv=8)
    return scores

def read_targets():
    """
    Read target data.

    Returns
    --------
    list
    """
    metadata = read_metadata()
    return metadata.target.values

def read_metadata():
    """
    Read metadata file.

    Returns
    --------
    dataframe
        Pandas df of metadata.
    """
    train_meta = pd.read_csv(_as_absolute('../raw_data/metadata_train.csv'))
    train_meta = train_meta[:TEST_CUTOFF]
    return train_meta

def read_data():
    """
    Read data file.

    Returns
    --------
    dataframe
        Pandas df of data.
    """
    train = pq.read_table(
            '../raw_data/train.parquet',
            columns=[str(i) for i in range(TEST_CUTOFF)]
            ).to_pandas()
    train = train.T
    return train


def _as_absolute(path):
    """
    Convert path to absolute from file called in.
    Parameters
    ----------
    path: str
        Path to make absolute.

    Returns
    --------
    str
        Absolute path.
    """
    return os.path.abspath(os.path.join(__file__, "../", path))



def _store_validation_result():
    pass

