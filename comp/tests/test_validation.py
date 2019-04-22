#!/usr/bin/env python3

"""
Tests for validation module
"""
import unittest
from unittest.mock import Mock, patch

import csv
import os
import shutil
from sklearn.base import BaseEstimator, ClassifierMixin

from comp import validation


def _as_absolute(path):
    return os.path.abspath(os.path.join(__file__, "../", path))

class TestValidationRunAndStore(unittest.TestCase):

    tmp_path = _as_absolute('tmp')

    def setUp(self):
        os.mkdir(self.tmp_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_path)

    def test_run_and_store(self):
        """
        Test full validation run and store result
        """
        store_path = 'tmp/test_store.csv'
        abs_store_path = _as_absolute(store_path)
        test_config1 = {
                'store_path':abs_store_path,
                'X':range(10),
                'y':[1]*10,
                'cv':5,
                }
        validation.set_config(test_config1)
        clf = Always1Classifier()
        clf.fit(None)
        validation.run(clf,'TestClassifier', '1', 'A test model')

        with open(abs_store_path, 'r') as f:
            reader = csv.reader(f)
            contents = [row for row in reader]
        print(contents)
        self.assertEqual(contents[0][0], 'model_name')
        self.assertEqual(contents[0][1], 'val_avg')
        self.assertEqual(contents[0][4], 'dt')
        self.assertEqual(contents[1][0], 'TestClassifier')
        self.assertEqual(contents[1][1], '1.0')
        self.assertEqual(contents[1][7], 'A test model')

        validation.run(clf,'TestClassifier', '1', 'A test model')

        with open(abs_store_path, 'r') as f:
            reader = csv.reader(f)
            contents = [row for row in reader]
        self.assertEqual(contents[2][0], 'TestClassifier')
        self.assertEqual(contents[2][1], '1.0')
        self.assertEqual(contents[2][7], 'A test model')
        os.remove(abs_store_path)


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

