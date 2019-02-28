#!/usr/bin/env python3

"""
Tests for validation module
"""
import unittest
from unittest.mock import Mock, patch

from comp import validation

from sklearn.base import BaseEstimator, ClassifierMixin

class Always1Classifier(ClassifierMixin, BaseEstimator):
    """ Estimator which always predicts 1 """

    def __init__(self, has_been_fit=False):
        self.has_been_fit = has_been_fit

    def fit(self, X, y=None):
        self.has_been_fit = True
        return self

    def predict(self, X):
        try:
            assert self.has_been_fit == True
        except AssertionError:
            raise RuntimeError("Must fit before predicting.")
        return [1 for x in X]

class TestBasicClassifier(unittest.TestCase):

    def test_score_all_1s(self):
        """
        Test validation runs and produces 1.0s for all 1s basic case.
        """

        clf = Always1Classifier()
        clf.fit(None)
        scores = validation._default_validaton(
                clf,
                X=range(10),
                y=[1]*10,
                cv=5,
                overide_config=True,
                )
        assert (scores[2] == [1.0]*5).all()

    def test_score_all_0s(self):
        """
        Test validation runs and produces 0.0s for all 0s basic case.
        """

        clf = Always1Classifier()
        clf.fit(None)
        scores = validation._default_validaton(
                clf,
                X=range(10),
                y=[0]*10,
                cv=5,
                overide_config=True,
                )
        assert (scores[2] == [0.0]*5).all()

