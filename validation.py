"""
Module for validation computation and validation results storing+reading
"""
import numpy as np
from sklearn import model_selection
from sklearn.metrics import matthews_corrcoef

# set random seed for reproduce
np.random.seed(41732)
# WARN: if using multithreading see https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener

# for scorers and other options see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

"""
Solution to config problem:
    Set the config on the module every time you use it. This works
    because that config can be written once in the _specific_ competion
    module code and passed in one line each time. And thus tracks the
    config, ties it to the specific competition code, and keeps things
    clean, ESPECIALLY clean because it means all the paths to data and
    the store etc are only defined in one place and can make those
    absolute so that you can run val from any depth location and still
    works. Nice.
"""

"""
Justification for using `global` in this module:
    The validation, and it's config, should be a singleton. Modules are
    singletons - using a global var in a module makes that var a singleton,
    hence I think this is a legitimate use of global.
    Note that the global is only at module level, not programme level, so you
    can't access `default_validation_config` without going through
    `comp.validation.defau...`.

"""


"""
Object to completely specify a default (k-fold) validation, including metric and
data etc.
Currently this is args for `model_selection.cross_val_score`, but could be made
to enable an entirely custom val function to be plugged in, depending on the
comp/aim. Should probably move to having it be plugged in, but good to have
a default. And, should not contain any config specific to the model being
validated.
"""
DEFAULT_VALIDATION_DEFAULT_CONFIG = {
        'store_path':None,
        'X':None,
        'y':None,
        'groups':None,
        'scoring':None,
        'cv':None,
        }

default_validation_config = {}

def reset_config():
    """
    Reset default validation config.

    Returns
    --------
    None

    Sets
    --------
    `default_validation_config`

    """
    default_validation_config = None


def set_config(config):
    """
    Set config for default validation function - k-fold.

    Parameters
    ----------
    config: dict
        Configuration dictionary used for validation setup.

    Returns
    --------
    None

    Sets
    --------
    `default_validation_config` with new config values.

    """
    global default_validation_config
    default_validation_config = DEFAULT_VALIDATION_DEFAULT_CONFIG
    for key in config:
        if key in default_validation_config:
            default_validation_config[key] = config[key]
        else:
            raise ValueError(f'Bad config key given: {key}')
    return default_validation_config

def _default_validaton(model,
        model_hyperparams=None,
        X=None,
        y=None,
        groups=None,
        scoring=None,
        cv=5,
        overide_config=False,
        ):
    """
    Default validation function - k-fold.

    Parameters
    ----------
    model: estimator object implementing ‘fit’
        Must follow sklearn `BaseEstimator` api.
        (I.e. implement `fit` and `predict` functions.)
        The object to use to fit the data.
    model_hyperparams: dict
        Optional - dict of hyperparameters
    X: array-like
        Optional - The data to fit. Can be for example a list, or an array.
    y: array-like, optional, default: None
        Optional - The target variable to try to predict.
    groups: array-like, with shape (n_samples,), optional
        Optional - Group labels for the samples used while splitting the dataset
        into train/test set.
    scoring: string, callable or None, optional, default: None
        Optional - A string (see scikit-learn model evaluation documentation) or
        a scorer callable object/function with signature
        `scorer(estimator, X, y)`.
    cv: int
        Optional - Specify the number of folds in a (Stratified)KFold,
    overide_config: bool
        Optional - Flag to choose whether to use the packages config for val
        setup or to specify setup explicitly as functions params. If `True` and
        config params are not `None` exception will be thrown.

    Returns
    --------
    Array of: [val_avg, val_std, list[raw validation scores]]

    """
    if overide_config:
        scores = model_selection.cross_val_score(
                model,
                X,
                y=y,
                groups=groups,
                scoring=scoring,
                cv=cv,
                fit_params=model_hyperparams,
                )
    return np.mean(scores), np.std(scores), scores

custom_validation = None

def set_custom_validation():
    pass

def _custom_validation():
    pass

def _store_validation_result(
        store_path,
        dt,
        elapsed_time,
        model_name,
        model_version,
        model_desc,
        model_fit_params,
        git_sha,
        val_avg,
        val_std,
        val_raw,
        ):
    """
    Stores info on validation run.

    Parameters
    ----------
    store_path: str/path
        Absolute (from here) path to store file.
    dt: datetime
        Datetime of validation run (start).
    elapsed_time: timedelta
        Time taken for validation to run.
    model_name: str
        Evocative/brief name of model. Likely the name of the model class.
    model_version: str
        Version num for use on minor variations of a model.
    model_desc: str
        Extended description of model.
    model_fit_params: dict
        Parameters that were passed to the fit method of the estimator. Can be
        empty.
    git_sha: str
        The sha of the current git commit. Note that `run` should throw if
        untracked changes present in model code.
    val_avg: float
        Average score over cross-validations folds.
    val_std: float
        Standard deviation of scores over cross-validations folds.
    val_raw: list[float]
        List of floats for raw scores of each cross-validation fold.

    Returns
    --------
    None

    Writes
    -------
    Validation result `store_path`.

    """
    pass

