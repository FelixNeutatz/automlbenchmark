import logging
import os
import pprint
import sys
import tempfile as tmp

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstructionScikit import ComplexityDrivenFeatureConstructionScikit

import numpy as np

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


from automl.benchmark import TaskConfig
from automl.data import Dataset
from automl.datautils import Encoder, impute, write_csv
from automl.results import save_predictions_to_file
from automl.utils import Timer, split_path, path_from_split


log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
    log.info("\n**** Complexity-Driven Feature Construction ****\n")

    is_classification = config.type == 'classification'
    # Mapping of benchmark metrics to TPOT metrics
    metrics_mapping = dict(
        acc=make_scorer(accuracy_score),
        auc=make_scorer(roc_auc_score),
        f1=make_scorer(f1_score),
        logloss=make_scorer(log_loss, greater_is_better=False, needs_proba=True),
        #mae='neg_mean_absolute_error',
        #mse='neg_mean_squared_error',
        #msle='neg_mean_squared_log_error',
        r2=make_scorer(r2_score)
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info('Running Complexity-Driven Feature Construction with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)

    #model = LogisticRegression
    #parameter_grid = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'], 'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class':['auto']}
    #parameter_grid = {'penalty': ['l2'], 'solver': ['lbfgs'],'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class': ['auto']}

    from sklearn.svm import SVC
    model = SVC
    parameter_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': ['balanced'], 'probability': [True], 'kernel': ['linear']}

    '''
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier
    parameter_grid = {'n_neighbors': np.arange(3, 10),
                                                                           'weights': ['uniform', 'distance'],
                                                                           'metric': ['minkowski', 'euclidean',
                                                                                      'manhattan']}
    '''


    if not is_classification:
        model = LinearRegression
        parameter_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

    is_categorical = [True if p.is_categorical() else False for p in dataset.predictors]
    names = [p.name for p in dataset.predictors]

    #fe = ComplexityDrivenFeatureConstructionScikit(max_time_secs=config.max_runtime_seconds, scoring=scoring_metric, model=model, parameter_grid=parameter_grid, n_jobs=n_jobs, epsilon=-np.inf)
    fe = ComplexityDrivenFeatureConstructionScikit(max_time_secs=config.max_runtime_seconds, scoring=scoring_metric,
                                                   model=model, parameter_grid=parameter_grid, n_jobs=n_jobs,
                                                   epsilon=0.0, feature_names=names, feature_is_categorical=is_categorical)

    with Timer() as training:
        fe.fit(dataset.train.X_enc, dataset.train.y_enc)

    log.info('Predicting on the test set.')
    predictions = fe.predict(dataset.test.X_enc)
    probabilities = None
    try:
        probabilities = fe.predict_proba(dataset.test.X_enc) if is_classification else None
    except RuntimeError:
        pass

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=dataset.test.y_enc,
                             target_is_encoded=is_classification)

    return dict(
        models_count=1, #todo
        training_duration=training.duration
    )
