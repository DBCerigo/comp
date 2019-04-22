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

class TestSetupConfigForDefaultValidation(unittest.TestCase):

    def test_basic_config_set(self):
        """
        Test that setting config correctly overwrites package config var.
        """
        test_config1 = {
                'store_path':'some/test/path',
                'X':'testData'
                }

        validation.set_config(test_config1)
        print(validation.default_validation_config)
        assert (validation.default_validation_config['store_path']
                == test_config1['store_path'])
        assert (validation.default_validation_config['X']
                == test_config1['X'])


    def test_set_bad_config_throws(self):
        """
        Test that setting config with bad key throws.
        """
        test_config2 = {
                'store_path':'some/test/path',
                'bad_key':'foo'
                }

        with self.assertRaises(ValueError):
            validation.set_config(test_config2)

class TestBasicClassifierWithConfig(unittest.TestCase):

    def test_score_all_1s(self):
        """
        Test validation runs and produces 1.0s for all 1s basic case with config
        """

        test_config1 = {
                'store_path':None,
                'X':range(10),
                'y':[1]*10,
                'cv':5,
                }
        validation.set_config(test_config1)
        clf = Always1Classifier()
        clf.fit(None)
        scores = validation._default_validaton(clf)
        assert (scores[2] == [1.0]*5).all()

