#!/usr/bin/env python3
"""
Module for validation computation and validation results storing+reading
"""
import numpy as np
from sklearn import model_selection


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
    None

    Writes
    -------
    Validation results to `/results`
    """
    scores = cross_val_score(model, None, None, cv=8)

def _store_validation_result():
    pass

