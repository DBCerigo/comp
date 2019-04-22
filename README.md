# comp
Helper package for data science, specifically model iteration, specifically specifically Kaggle competitions

Implements API to run and store validation runs, and set config for comp/certain validation.

Usage: 
```
some_kaggle_comp_module.py 

config = {
    'store_path':'some_absolute_path',
     'X':data,
      'y':labels,
      'scoring':some_scorer,
      'cv':8}
```

```
some_notebook.py 

import some_kaggle_comp_module
from comp import validation
validation.set_config(some_kaggle_comp_module.config)

... make a model ...

validation.run(model, model_name, model_version, model_description)
```

Stores:
```
'model_name', 'val_avg', 'val_std', 'val_raw', 'dt', 'elapsed_time', 'model_version', 'model_desc', 'fit_params', '__class__', 'git_sha', 'validation_config'
'TestClassifier', '1.0', '0.0', '[1. 1. 1. 1. 1.]', '2019-04-22 22:47:04.174894', '0:00:00.014047', '1', 'A test model', '', "<class 'comp.tests.test_validation.Always1Classifier'>", 'f5878fc2fc01bc6620e9e78d00fecc13b5eef17b', "{'store_path': '/Users/dbcerigo/dev/powerlinefault/comp/tests/tmp/test_store.csv', 'X': range(0, 10), 'y': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'groups': None, 'scoring': None, 'cv': 5}"
```
