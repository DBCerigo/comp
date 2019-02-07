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


# for scorers and other options see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

def run(model,
        X,
        y=None,
        groups=None,
        scoring=None,
        cv=5,
        model_hyperparams=None,
        ):
    """
    Main validation function.

    Parameters
    ----------
    model: estimator object implementing ‘fit’
        Must follow sklearn `BaseEstimator` api.
        (I.e. implement `fit` and `predict` functions.)
        The object to use to fit the data.
    X: array-like
        The data to fit. Can be for example a list, or an array.
    y: array-like, optional, default: None
        The target variable to try to predict.
    groups: array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
    cv: int
        Specify the number of folds in a (Stratified)KFold,
    model_hyperparams: dict
        Optional dict of hyperparameters

    Returns
    --------
    Array of validation scores.

    Writes
    -------
    Validation results to `/results`.
    """
    scores = cross_val_score(
            model,
            X,
            y=y,
            groups=groups,
            scoring=scoring,
            cv=cv)
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

