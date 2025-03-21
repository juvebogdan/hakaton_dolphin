"""
config.py

Configuration parameters for the dolphin detector.
"""

# Audio Processing Parameters
SAMPLE_RATE = 96000  # Hz
MAX_FREQUENCY = 320  # Maximum frequency slice for time stats

# Filtering Parameters
FILTER_LOW = 5000   # Low cutoff frequency (Hz)
FILTER_HIGH = 15000  # High cutoff frequency (Hz)

# XGBoost Default Parameters
XGB_PARAMS = {
    'max_depth': 5,
    'subsample': 0.8,
    'n_estimators': 100,
    'learning_rate': 0.05,
    'min_child_weight': 5,
    'colsample_bytree': 0.8,
    'gamma': 0.1
}

# XGBoost Parameter Grid for Tuning
XGB_PARAM_GRID = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__gamma': [0, 0.1, 0.2]
}

# Cross-validation Parameters
CV_FOLDS = 10
SMALL_TEST_CV_FOLDS = 3

# Data Split Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paths
OUTPUT_DIR = 'output'
TEMPLATE_DIR = 'templates'

# Visualization Parameters
FIGURE_SIZE = (12, 8)
DPI = 100
