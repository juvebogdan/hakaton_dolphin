"""
enhanced_dolphin_classifier.py

An enhanced version of the dolphin classifier with advanced parameter tuning
and overfitting prevention techniques that can be toggled on/off.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, 
                                    cross_validate, GridSearchCV, RandomizedSearchCV,
                                    learning_curve, validation_curve, KFold)
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                            precision_recall_curve, accuracy_score, roc_auc_score, 
                            f1_score, precision_score, recall_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Optional imports (will be checked before use)
OPTUNA_AVAILABLE = False
SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# Add the parent directory to sys.path to allow both package and script usage
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from dolphin_detector import config
except ImportError:
    # If config module not available, create a minimal config
    class ConfigClass:
        def __init__(self):
            self.RANDOM_STATE = 42
            self.TEST_SIZE = 0.2
            self.MAX_FREQUENCY = 150
            self.CV_FOLDS = 5
            self.SMALL_TEST_CV_FOLDS = 3
            self.FIGURE_SIZE = (10, 8)
            self.DPI = 300
            self.OUTPUT_DIR = "output"
            self.XGB_PARAMS = {
                'max_depth': 5,
                'subsample': 0.8,
                'n_estimators': 100,
                'learning_rate': 0.05,
                'min_child_weight': 5,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            self.XGB_PARAM_GRID = {
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__subsample': [0.7, 0.8, 0.9],
                'classifier__colsample_bytree': [0.7, 0.8, 0.9],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__gamma': [0, 0.1, 0.2]
            }
    config = ConfigClass()


class EnhancedDolphinClassifier:
    """
    Enhanced classifier for dolphin whistle detection with advanced
    parameter tuning and overfitting prevention techniques.
    """
    
    def __init__(self, metrics_file, random_state=config.RANDOM_STATE, 
                 threshold_optimization=False, skip_time_metrics=True):
        """
        Initialize the enhanced classifier
        
        Args:
            metrics_file: Path to the metrics CSV file
            random_state: Random seed for reproducibility
            threshold_optimization: Whether to use threshold optimization
            skip_time_metrics: Whether to skip time metrics in feature extraction
        """
        self.metrics_file = metrics_file
        self.random_state = random_state
        self.threshold_optimization = threshold_optimization
        self.skip_time_metrics = skip_time_metrics
        self.load_data()
        self.scaler = StandardScaler()
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.dirname(self.metrics_file)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """
        Load the metrics data from CSV file and extract features
        If skip_time_metrics is True, exclude time metrics (centTime, bwTime, skewTime, tvTime)
        """
        print(f"Loading metrics from {self.metrics_file}")
        data = pd.read_csv(self.metrics_file)
        
        # Extract truth and index
        self.truth = np.array(data.iloc[:, 0])
        self.index = np.array(data.iloc[:, 1])
        
        # Get number of templates from the first prefix group (max_)
        n_templates = sum(1 for col in data.columns if col.startswith('max_'))
        
        # Calculate indices for different feature groups
        template_cols = []
        # Template metrics (max, xLoc, yLoc) for both vertical and horizontal enhancement
        for prefix in ['max', 'xLoc', 'yLoc']:
            # For vertical enhancement
            template_cols.extend([i for i, col in enumerate(data.columns) 
                               if col.startswith(f'{prefix}_')])
            # For horizontal enhancement
            template_cols.extend([i for i, col in enumerate(data.columns) 
                               if col.startswith(f'{prefix}H_')])
        
        # Find where time metrics end and placeholder metrics begin
        time_metrics_end = 2  # Skip truth and index
        for prefix in ['max', 'xLoc', 'yLoc']:
            time_metrics_end += 2 * n_templates  # Both vertical and horizontal
        
        if self.skip_time_metrics:
            # Skip time metrics (centTime, bwTime, skewTime, tvTime)
            time_metrics_end += 4 * config.MAX_FREQUENCY
        else:
            # Include time metrics columns
            time_metrics = []
            for prefix in ['centTime', 'bwTime', 'skewTime', 'tvTime']:
                time_metrics.extend([i for i, col in enumerate(data.columns) 
                                   if col.startswith(prefix)])
            template_cols.extend(time_metrics)
        
        # Get remaining columns (placeholder and high freq metrics)
        remaining_cols = list(range(time_metrics_end, len(data.columns)))
        
        # Combine all feature columns
        feature_cols = sorted(set(template_cols + remaining_cols))
        
        # Extract features
        self.features = np.array(data.iloc[:, feature_cols])
        
        # Create feature names
        self.feature_names = []
        # Template metrics names
        for enhancement in ['', 'H_']:  # '' for vertical, 'H_' for horizontal
            for prefix in ['max', 'xLoc', 'yLoc']:
                for i in range(n_templates):
                    self.feature_names.append(f"{prefix}{enhancement}template_{i+1}")
        
        if not self.skip_time_metrics:
            # Add time metric names if included
            for prefix in ['centTime', 'bwTime', 'skewTime', 'tvTime']:
                for i in range(config.MAX_FREQUENCY):
                    self.feature_names.append(f"{prefix}_{i:04d}")
        
        # Add placeholder metric names
        self.feature_names.extend([f'placeholder_{i:04d}' for i in range(50)])
        
        # Add high frequency metric names
        self.feature_names.extend([
            'CentStd',   # Standard deviation of centroids
            'AvgBwd',    # Mean bandwidth
            'hfCent',    # High-frequency centroid
            'hfBwd',     # High-frequency bandwidth
            'hfMax',     # Bar template matching score
            'hfMax2',    # Bar1 template matching score
            'hfMax3'     # Bar2 template matching score
        ])
        
        # Print dataset statistics
        self.n_samples, self.n_features = self.features.shape
        n_positive = np.sum(self.truth == 1)
        n_negative = np.sum(self.truth == 0)
        
        print(f"Loaded {self.n_samples} samples with {self.n_features} features")
        print(f"Positive samples (whistles): {n_positive}")
        print(f"Negative samples (noise): {n_negative}")
        print(f"Time metrics: {'excluded' if self.skip_time_metrics else 'included'}")
    
    def prepare_data(self, test_size=config.TEST_SIZE, val_size=0.1, small_test=False):
        """
        Prepare data for training, validation, and testing
        
        Args:
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            small_test: If True, use only a small subset of data for quick testing
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Data splits
        """
        # Check for NaN values
        nan_count = np.isnan(self.features).sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in the dataset")
        
        # Use small subset for testing if requested
        if small_test:
            # Use only 10% of the data for quick testing
            test_sample_size = min(500, int(self.n_samples * 0.1))
            print(f"Using small test mode with {test_sample_size} samples")
            
            # Ensure balanced classes in the small test
            pos_indices = np.where(self.truth == 1)[0]
            neg_indices = np.where(self.truth == 0)[0]
            
            # Sample equal numbers from each class
            sample_size_per_class = test_sample_size // 2
            pos_sample = np.random.choice(pos_indices, size=sample_size_per_class, replace=False)
            neg_sample = np.random.choice(neg_indices, size=sample_size_per_class, replace=False)
            
            # Combine samples
            sample_indices = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(sample_indices)
            
            # Extract subset
            features_subset = self.features[sample_indices]
            truth_subset = self.truth[sample_indices]
            index_subset = self.index[sample_indices]
            
            # Store original data
            self._original_features = self.features
            self._original_truth = self.truth
            self._original_index = self.index
            
            # Replace with subset
            self.features = features_subset
            self.truth = truth_subset
            self.index = index_subset
            self.n_samples = len(self.truth)
            
            print(f"Small test dataset: {self.n_samples} samples")
            print(f"Positive samples: {np.sum(self.truth == 1)}")
            print(f"Negative samples: {np.sum(self.truth == 0)}")
        
        # First split: training+validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.features, self.truth, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=self.truth
        )
        
        # Second split: training vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size / (1 - test_size),  # Adjust val_size relative to train_val size
            random_state=self.random_state,
            stratify=y_train_val
        )
        
        print(f"Data split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/self.n_samples:.1%})")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/self.n_samples:.1%})")
        print(f"  Test:       {len(X_test)} samples ({len(X_test)/self.n_samples:.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_with_early_stopping(self, early_stopping=True):
        """
        Train the model with optional early stopping
        
        Args:
            early_stopping: Whether to use early stopping
            
        Returns:
            Trained model and accuracy
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Define XGBoost parameters
        xgb_params = config.XGB_PARAMS.copy()
        
        if early_stopping:
            print("\nUsing early stopping with validation set")
            # Define classifier with early stopping
            xgb_clf = xgb.XGBClassifier(
                **xgb_params,
                early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
                random_state=self.random_state
            )
            
            # Create pipeline for the classifier
            xgb_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', xgb_clf)
            ])
            
            # Train with validation monitoring
            print("Training XGBoost classifier with early stopping...")
            xgb_pipeline.fit(
                X_train, y_train,
                classifier__eval_set=[(X_val, y_val)],
                classifier__eval_metric='auc'  # Use appropriate metric
            )
            
            # Get best iteration
            best_iteration = xgb_clf.best_iteration
            print(f"Best iteration: {best_iteration}")
        else:
            # Create pipeline without early stopping
            xgb_clf = xgb.XGBClassifier(**xgb_params, random_state=self.random_state)
            xgb_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', xgb_clf)
            ])
            
            print("Training XGBoost classifier without early stopping...")
            xgb_pipeline.fit(X_train, y_train)
        
        # Evaluate model
        xgb_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
        
        if self.threshold_optimization:
            # Find and use optimal threshold
            optimal_threshold = self.find_optimal_threshold(y_test, xgb_pred_proba)
            xgb_pred = (xgb_pred_proba >= optimal_threshold).astype(int)
            print(f"\nXGBoost accuracy with optimal threshold: {accuracy_score(y_test, xgb_pred):.4f}")
            self.optimal_threshold = optimal_threshold
        else:
            # Use default threshold (0.5)
            xgb_pred = xgb_pipeline.predict(X_test)
            print(f"\nXGBoost accuracy with default threshold (0.5): {accuracy_score(y_test, xgb_pred):.4f}")
            self.optimal_threshold = 0.5
        
        # Store predictions and model
        self.xgb_pipeline = xgb_pipeline
        self.y_pred_proba = xgb_pred_proba
        self.y_pred = xgb_pred
        self.y_test = y_test
        self.X_test = X_test
        
        # For feature importance
        self.clf = xgb_pipeline.named_steps['classifier']
        
        # Calculate additional metrics
        accuracy = accuracy_score(y_test, xgb_pred)
        auc_roc = roc_auc_score(y_test, xgb_pred_proba)
        f1 = f1_score(y_test, xgb_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return xgb_pipeline, accuracy
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Find the optimal classification threshold using ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            float: Optimal threshold value
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Calculate the geometric mean of sensitivity and specificity
        geometric_mean = np.sqrt(tpr * (1-fpr))
        
        # Find the optimal threshold
        optimal_idx = np.argmax(geometric_mean)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        optimal_accuracy = accuracy_score(y_true, y_pred_optimal)
        optimal_auc = roc_auc_score(y_true, y_pred_proba)
        
        print("\nOptimal Threshold Analysis:")
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Accuracy at optimal threshold: {optimal_accuracy:.4f}")
        print(f"AUC-ROC: {optimal_auc:.4f}")
        print(f"True Positive Rate: {tpr[optimal_idx]:.4f}")
        print(f"False Positive Rate: {fpr[optimal_idx]:.4f}")
        
        return optimal_threshold
    
    def tune_with_optuna(self, n_trials=100, small_test=False):
        """
        Tune parameters using Bayesian optimization with Optuna
        
        Args:
            n_trials: Number of trials for optimization
            small_test: If True, use small subset of data for quick testing
            
        Returns:
            Best parameters, best pipeline
        """
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna is not installed. Please install with 'pip install optuna'")
            return None, None
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(small_test=small_test)
        
        print(f"\n=== Bayesian Optimization with Optuna ({n_trials} trials) ===")
        
        # Define the objective function for Optuna
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.01, 5.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'random_state': self.random_state
            }
            
            clf = xgb.XGBClassifier(**param)
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            
            # Create early stopping callback
            callbacks = None
            if not small_test:
                # Use early stopping with validation set when not in small test mode
                pipeline.fit(
                    X_train, y_train,
                    classifier__eval_set=[(X_val, y_val)],
                    classifier__eval_metric='auc',
                    classifier__early_stopping_rounds=10,
                    classifier__verbose=False
                )
            else:
                # Simple fit for small test mode
                pipeline.fit(X_train, y_train)
            
            # Score on validation set
            y_pred = pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        # Create the study
        study = optuna.create_study(direction='maximize', 
                                   study_name='dolphin_whistle_classifier')
        
        # Optimize with progress bar
        print("Optimizing parameters...")
        study.optimize(objective, n_trials=n_trials)
        
        # Print results
        print("\nBest Optuna Parameters:")
        best_params = study.best_params
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best accuracy: {study.best_value:.4f}")
        
        # Create and train model with best parameters
        xgb_best = xgb.XGBClassifier(**best_params, random_state=self.random_state)
        best_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb_best)
        ])
        
        # Train on train+val data
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.hstack((y_train, y_val))
        
        print("Training final model with best parameters...")
        best_pipeline.fit(X_train_val, y_train_val)
        
        # Evaluate on test set
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        if self.threshold_optimization:
            # Find and use optimal threshold
            optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            self.optimal_threshold = optimal_threshold
        else:
            # Use default threshold (0.5)
            y_pred = best_pipeline.predict(X_test)
            self.optimal_threshold = 0.5
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nTest Set Performance with Best Parameters:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Store predictions and model
        self.xgb_pipeline = best_pipeline
        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_test = X_test
        
        # For feature importance
        self.clf = best_pipeline.named_steps['classifier']
        
        # Store best parameters
        self.best_params = best_params
        
        # Try to visualize parameter importance if in interactive mode
        if not small_test:
            try:
                param_importance_plot = optuna.visualization.plot_param_importances(study)
                print("Parameter importance visualization generated. Check Optuna visualization.")
            except Exception as e:
                print(f"Could not generate parameter importance visualization: {e}")
                
        return best_params, best_pipeline
    
    def nested_cross_validation(self, outer_cv=5, inner_cv=3, small_test=False):
        """
        Perform nested cross-validation for unbiased performance estimation
        
        Args:
            outer_cv: Number of folds for outer CV
            inner_cv: Number of folds for inner CV
            small_test: If True, use a small subset of data for quick testing
            
        Returns:
            Nested CV scores
        """
        if small_test:
            outer_cv = min(outer_cv, 3)
            inner_cv = min(inner_cv, 2)
        
        print(f"\n=== Nested Cross-Validation ({outer_cv}x{inner_cv}) ===")
        
        # Prepare data (without splitting)
        X, y = self.features, self.truth
        
        if small_test:
            # Use only 10% of the data
            test_sample_size = min(500, int(self.n_samples * 0.1))
            
            # Sample stratified subset
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            sample_size_per_class = test_sample_size // 2
            pos_sample = np.random.choice(pos_indices, size=sample_size_per_class, replace=False)
            neg_sample = np.random.choice(neg_indices, size=sample_size_per_class, replace=False)
            
            sample_indices = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(sample_indices)
            
            X = X[sample_indices]
            y = y[sample_indices]
        
        # Define CV splits
        outer_cv_split = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        inner_cv_split = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state)
        
        # Parameter grid (simplified if small_test)
        if small_test:
            param_grid = {
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__n_estimators': [50, 100],
                'classifier__subsample': [0.8],
                'classifier__min_child_weight': [1, 5]
            }
        else:
            param_grid = {
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__subsample': [0.7, 0.8, 0.9],
                'classifier__colsample_bytree': [0.7, 0.8, 0.9],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__gamma': [0, 0.1, 0.2],
                'classifier__reg_alpha': [0, 0.1, 1.0],
                'classifier__reg_lambda': [0, 0.1, 1.0]
            }
        
        # Create the base classifier
        base_classifier = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(random_state=self.random_state))
        ])
        
        # Initialize metrics lists
        nested_scores = []
        best_params_list = []
        fold_metrics = []
        
        print("Performing nested cross-validation...")
        # Outer CV loop
        for i, (train_idx, test_idx) in enumerate(tqdm(outer_cv_split.split(X, y), 
                                                     total=outer_cv, 
                                                     desc="Outer CV")):
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV for parameter tuning
            clf = GridSearchCV(
                estimator=base_classifier,
                param_grid=param_grid,
                cv=inner_cv_split,
                scoring='accuracy',
                n_jobs=-1 if not small_test else 1  # Parallel processing if not in small test mode
            )
            
            # Train on inner CV
            clf.fit(X_train_outer, y_train_outer)
            
            # Get best parameters
            best_params = clf.best_params_
            best_params_list.append(best_params)
            
            # Evaluate on outer test set
            y_pred_proba = clf.predict_proba(X_test_outer)[:, 1]
            y_pred = clf.predict(X_test_outer)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_outer, y_pred)
            auc_score = roc_auc_score(y_test_outer, y_pred_proba)
            f1 = f1_score(y_test_outer, y_pred)
            
            # Store metrics
            nested_scores.append(accuracy)
            fold_metrics.append({
                'accuracy': accuracy,
                'auc': auc_score,
                'f1': f1,
                'best_params': best_params
            })
            
            print(f"  Fold {i+1} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
        
        # Calculate mean and std of nested CV score
        print("\nNested CV Results:")
        print(f"Accuracy: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
        
        # Summarize metrics across folds
        metrics_df = pd.DataFrame([{
            'fold': i+1,
            'accuracy': m['accuracy'],
            'auc': m['auc'],
            'f1': m['f1']
        } for i, m in enumerate(fold_metrics)])
        
        print("\nMetrics Summary Across Folds:")
        print(metrics_df.describe().loc[['mean', 'std', 'min', 'max']].round(4))
        
        # Analyze parameter stability
        param_counts = {}
        for params in best_params_list:
            for param, value in params.items():
                if param not in param_counts:
                    param_counts[param] = {}
                if value not in param_counts[param]:
                    param_counts[param][value] = 0
                param_counts[param][value] += 1
        
        print("\nParameter Stability Analysis:")
        for param, counts in param_counts.items():
            print(f"  {param.replace('classifier__', '')}:")
            for value, count in counts.items():
                print(f"    {value}: {count}/{outer_cv} folds")
        
        # Plot nested CV results
        self.plot_nested_cv_results(metrics_df)
        
        # Store nested CV results
        self.nested_cv_results = {
            'scores': nested_scores,
            'fold_metrics': fold_metrics,
            'param_counts': param_counts
        }
        
        return nested_scores, fold_metrics
    
    def plot_nested_cv_results(self, metrics_df):
        """
        Plot nested CV results
        
        Args:
            metrics_df: DataFrame with metrics for each fold
        """
        plt.figure(figsize=(12, 8))
        
        # Plot metrics distribution
        plt.subplot(2, 1, 1)
        metrics_melted = pd.melt(metrics_df, id_vars=['fold'], 
                                value_vars=['accuracy', 'auc', 'f1'],
                                var_name='metric', value_name='value')
        
        sns.boxplot(x='metric', y='value', data=metrics_melted)
        sns.stripplot(x='metric', y='value', data=metrics_melted, 
                     color='black', alpha=0.5, jitter=True)
        
        plt.title('Metrics Distribution Across Folds')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Plot metrics by fold
        plt.subplot(2, 1, 2)
        metrics_df.set_index('fold')[['accuracy', 'auc', 'f1']].plot(marker='o')
        plt.title('Metrics by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(metrics_df['fold'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'nested_cv_results.png'))
        plt.close()
        
        print(f"Nested CV results plot saved to {os.path.join(self.output_dir, 'nested_cv_results.png')}")
    
    def plot_learning_curves(self, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
        """
        Plot learning curves to diagnose bias/variance issues
        
        Args:
            train_sizes: Array of training set sizes to plot
            cv: Number of cross-validation folds
            
        Returns:
            Training and test scores for different training set sizes
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Combine train and validation sets
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.hstack((y_train, y_val))
        
        print("\n=== Generating Learning Curves ===")
        print("This may take a while...")
        
        # Create pipeline
        xgb_params = config.XGB_PARAMS.copy()
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(**xgb_params, random_state=self.random_state))
        ])
        
        # Calculate learning curve
        train_sizes_abs, train_scores, test_scores = learning_curve(
            xgb_pipeline, X_train_val, y_train_val, 
            train_sizes=train_sizes,
            cv=cv, scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Calculate mean and std of scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes_abs, test_mean, 'o-', color='g', label='Cross-validation score')
        
        # Add std deviation areas
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.title('Learning Curves')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add gap between training and CV scores
        plt.ylim([min(train_mean.min(), test_mean.min()) * 0.95, 
                max(train_mean.max(), test_mean.max()) * 1.02])
        
        # Add interpretation
        gap = train_mean - test_mean
        mean_gap = np.mean(gap[-3:])  # Average gap for last 3 points
        
        if mean_gap > 0.1:
            plt.text(0.5, 0.1, "High variance (overfitting)", 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(facecolor='red', alpha=0.1))
        elif mean_gap < 0.05 and train_mean[-1] < 0.85:
            plt.text(0.5, 0.1, "High bias (underfitting)", 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(facecolor='blue', alpha=0.1))
        else:
            plt.text(0.5, 0.1, "Good balance", 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(facecolor='green', alpha=0.1))
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'))
        plt.close()
        
        print(f"Learning curves saved to {os.path.join(self.output_dir, 'learning_curves.png')}")
        
        # Analyze learning curve
        print("\nLearning Curve Analysis:")
        print(f"Final training score: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
        print(f"Final CV score: {test_mean[-1]:.4f} ± {test_std[-1]:.4f}")
        print(f"Gap between training and CV scores: {mean_gap:.4f}")
        
        if mean_gap > 0.1:
            print("Diagnosis: High variance (overfitting)")
            print("Recommendations:")
            print("  - Increase regularization (higher alpha/lambda)")
            print("  - Reduce model complexity (lower max_depth)")
            print("  - Add more training data")
            print("  - Feature selection/reduction")
        elif mean_gap < 0.05 and train_mean[-1] < 0.85:
            print("Diagnosis: High bias (underfitting)")
            print("Recommendations:")
            print("  - Increase model complexity (higher max_depth)")
            print("  - Reduce regularization")
            print("  - Add more features or feature interactions")
            print("  - Consider more powerful models")
        else:
            print("Diagnosis: Good balance between bias and variance")
            
        # Store learning curve results
        self.learning_curve_results = {
            'train_sizes_abs': train_sizes_abs,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std
        }
        
        return train_mean, test_mean
    
    def select_features(self, threshold=0.01, use_shap=False):
        """
        Select features based on importance threshold
        
        Args:
            threshold: Importance threshold for feature selection
            use_shap: Whether to use SHAP values for feature selection
            
        Returns:
            Selected feature indices, selector
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        print("\n=== Feature Selection ===")
        
        # Train an initial model to get feature importances
        xgb_clf = xgb.XGBClassifier(**config.XGB_PARAMS, random_state=self.random_state)
        
        # Create pipeline with preprocessing
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb_clf)
        ])
        
        # Train the model
        print("Training model for feature selection...")
        xgb_pipeline.fit(X_train, y_train)
        
        # Get classifier from pipeline
        clf = xgb_pipeline.named_steps['classifier']
        
        # Get feature importances - using SHAP or built-in method
        if use_shap and SHAP_AVAILABLE:
            print("Using SHAP values for feature importance...")
            
            # Transform data with preprocessing steps
            imputer = xgb_pipeline.named_steps['imputer']
            scaler = xgb_pipeline.named_steps['scaler']
            X_train_processed = scaler.transform(imputer.transform(X_train))
            
            # Create explainer
            explainer = shap.TreeExplainer(clf)
            
            # Calculate SHAP values (limit to 1000 samples for speed)
            sample_size = min(1000, len(X_train_processed))
            sample_indices = np.random.choice(len(X_train_processed), size=sample_size, replace=False)
            shap_values = explainer.shap_values(X_train_processed[sample_indices])
            
            # Get absolute mean of SHAP values as importance
            importances = np.abs(shap_values).mean(axis=0)
            

            
            # Ensure feature_names matches the actual number of features
            n_actual_features = X_train_processed.shape[1]
            if len(self.feature_names) < n_actual_features:
                # If we have fewer names than actual features in the data
                # (this can happen if some features were generated during preprocessing)
                print(f"Warning: Found {n_actual_features} features but only {len(self.feature_names)} feature names")
                # Extend feature names list with generic names for the additional features
                self.feature_names.extend([f'generated_feature_{i}' for i in range(len(self.feature_names), n_actual_features)])
            elif len(self.feature_names) > n_actual_features:
                # If we have more names than actual features
                # (this can happen if some features were dropped during preprocessing)
                print(f"Warning: Found {n_actual_features} features but have {len(self.feature_names)} feature names")
                print("Truncating feature names list to match actual features")
                self.feature_names = self.feature_names[:n_actual_features]
            
            print(f"Using {len(self.feature_names)} feature names for {n_actual_features} actual features")
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 10))
            try:
                shap.summary_plot(shap_values, X_train_processed[sample_indices], 
                                feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'))
                plt.close()
                print(f"SHAP summary plot saved to {os.path.join(self.output_dir, 'shap_summary.png')}")
            except Exception as e:
                print(f"Warning: Could not generate SHAP summary plot: {e}")
                plt.close()
        else:
            # Use built-in feature importance
            print("Using built-in feature importance...")
            importances = clf.feature_importances_
        
        # Normalize importances
        importances = importances / np.sum(importances)
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Print top features
        print("\nTop features by importance:")
        for i, idx in enumerate(indices[:20]):  # Top 20 features
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
            print(f"{i+1}. {feature_name}: {importances[idx]:.4f}")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        n_features_to_plot = min(30, len(indices))
        plt.bar(range(n_features_to_plot), importances[indices[:n_features_to_plot]], align='center')
        
        # Get feature names for plotting, using generic names if needed
        plot_feature_names = []
        for idx in indices[:n_features_to_plot]:
            if idx < len(self.feature_names):
                plot_feature_names.append(self.feature_names[idx])
            else:
                plot_feature_names.append(f'feature_{idx}')
        
        plt.xticks(range(n_features_to_plot), plot_feature_names, rotation=90)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importances.png'))
        plt.close()
        print(f"Feature importance plot saved to {os.path.join(self.output_dir, 'feature_importances.png')}")
        
        # Select features based on threshold
        if threshold > 0:
            # Calculate cumulative importance
            cumulative_importance = np.cumsum(importances[indices])
            
            # Find index where cumulative importance exceeds threshold
            threshold_idx = np.where(cumulative_importance >= threshold)[0][0] + 1
            
            # Get selected feature indices
            selected_indices = indices[:threshold_idx]
            
            print(f"\nSelected {len(selected_indices)} out of {len(self.feature_names)} features")
            print(f"These features account for {cumulative_importance[threshold_idx-1]:.2%} of total importance")
            
            # Create a selector
            selector = SelectFromModel(clf, threshold=importances[indices[threshold_idx-1]],
                                     prefit=True)
            
            # Transform data
            X_train_selected = selector.transform(X_train)
            X_val_selected = selector.transform(X_val)
            X_test_selected = selector.transform(X_test)
            
            print(f"Reduced feature dimension from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            
            # Train new model with selected features
            print("\nTraining model with selected features...")
            xgb_clf_selected = xgb.XGBClassifier(**config.XGB_PARAMS, random_state=self.random_state)
            
            xgb_pipeline_selected = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', xgb_clf_selected)
            ])
            
            xgb_pipeline_selected.fit(X_train_selected, y_train)
            
            # Evaluate model with selected features
            y_pred_selected = xgb_pipeline_selected.predict(X_test_selected)
            y_pred_proba_selected = xgb_pipeline_selected.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics
            accuracy_selected = accuracy_score(y_test, y_pred_selected)
            auc_selected = roc_auc_score(y_test, y_pred_proba_selected)
            
            # Compare with full feature set
            y_pred_full = xgb_pipeline.predict(X_test)
            y_pred_proba_full = xgb_pipeline.predict_proba(X_test)[:, 1]
            
            accuracy_full = accuracy_score(y_test, y_pred_full)
            auc_full = roc_auc_score(y_test, y_pred_proba_full)
            
            print("\nPerformance Comparison:")
            print(f"Full Feature Set - Accuracy: {accuracy_full:.4f}, AUC: {auc_full:.4f}")
            print(f"Selected Features - Accuracy: {accuracy_selected:.4f}, AUC: {auc_selected:.4f}")
            
            # Store feature selection results
            self.feature_selection_results = {
                'importances': importances,
                'indices': indices,
                'selected_indices': selected_indices,
                'selector': selector,
                'accuracy_full': accuracy_full,
                'accuracy_selected': accuracy_selected,
                'auc_full': auc_full,
                'auc_selected': auc_selected
            }
            
            # Store selected feature pipeline
            self.xgb_pipeline_selected = xgb_pipeline_selected
            
            return selected_indices, selector
        else:
            return indices, None
    
    def plot_validation_curves(self, param_list=None, log_scale=None):
        """
        Plot validation curves for specified parameters
        
        Args:
            param_list: List of parameters to create validation curves for
            log_scale: Dictionary of parameters that should use log scale
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Combine train and validation sets
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.hstack((y_train, y_val))
        
        # Default parameters if not specified
        if param_list is None:
            param_list = [
                ('max_depth', [1, 2, 3, 5, 7, 10, 15]),
                ('learning_rate', [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]),
                ('n_estimators', [10, 25, 50, 100, 200, 500]),
                ('min_child_weight', [1, 3, 5, 7, 10, 15]),
                ('gamma', [0, 0.1, 0.2, 0.3, 0.5, 1.0]),
                ('reg_alpha', [0, 0.01, 0.1, 1.0, 10.0, 100.0]),
                ('reg_lambda', [0, 0.01, 0.1, 1.0, 10.0, 100.0])
            ]
        
        # Default log scale parameters
        if log_scale is None:
            log_scale = {
                'learning_rate': True,
                'reg_alpha': True,
                'reg_lambda': True
            }
        
        print("\n=== Generating Validation Curves ===")
        
        # Create pipeline
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(**config.XGB_PARAMS, random_state=self.random_state))
        ])
        
        # Create output directory for validation curves
        validation_curves_dir = os.path.join(self.output_dir, 'validation_curves')
        os.makedirs(validation_curves_dir, exist_ok=True)
        
        # Generate validation curves for each parameter
        for param_name, param_range in param_list:
            print(f"Generating validation curve for {param_name}...")
            
            # Calculate validation curve
            train_scores, test_scores = validation_curve(
                xgb_pipeline, X_train_val, y_train_val,
                param_name=f'classifier__{param_name}',
                param_range=param_range,
                cv=5, scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Calculate mean and std of scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot validation curve
            plt.figure(figsize=(10, 6))
            plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
            plt.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
            
            # Add std deviation areas
            plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
            plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
            
            plt.title(f'Validation Curve for {param_name}')
            plt.xlabel(param_name)
            if param_name in log_scale and log_scale[param_name]:
                plt.xscale('log')
            plt.ylabel('Accuracy Score')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Add optimal value marker
            best_idx = np.argmax(test_mean)
            best_value = param_range[best_idx]
            best_score = test_mean[best_idx]
            
            plt.plot([best_value], [best_score], 'o', color='blue', markersize=10,
                    label=f'Optimal: {best_value}')
            plt.legend(loc='best')
            
            # Add informative annotation
            plt.annotate(f'Optimal value: {best_value}\nScore: {best_score:.4f}',
                        xy=(best_value, best_score),
                        xytext=(0.5, 0.5),
                        textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            # Save the plot
            plt.savefig(os.path.join(validation_curves_dir, f'validation_curve_{param_name}.png'))
            plt.close()
            
            # Store optimal value
            if not hasattr(self, 'optimal_params'):
                self.optimal_params = {}
            self.optimal_params[param_name] = best_value
            
            print(f"  Optimal {param_name}: {best_value} (score: {best_score:.4f})")
        
        print(f"Validation curves saved to {validation_curves_dir}")
        print("\nOptimal Parameter Values:")
        for param, value in self.optimal_params.items():
            print(f"  {param}: {value}")
        
        return self.optimal_params
    
    def create_model_ensemble(self):
        """
        Create an ensemble of different models for improved performance
        
        Returns:
            Ensemble pipeline, accuracy
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Combine train and validation sets
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.hstack((y_train, y_val))
        
        print("\n=== Creating Model Ensemble ===")
        
        # Load optimized parameters if available, otherwise use defaults
        xgb_params = getattr(self, 'best_params', config.XGB_PARAMS.copy())
        
        # Create individual classifiers
        xgb_clf = xgb.XGBClassifier(**xgb_params, random_state=self.random_state)
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        lr_clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_clf),
                ('rf', rf_clf),
                ('lr', lr_clf)
            ],
            voting='soft'  # Use probability estimates
        )
        
        # Create pipeline
        ensemble_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', ensemble)
        ])
        
        # Train ensemble
        print("Training ensemble model...")
        ensemble_pipeline.fit(X_train_val, y_train_val)
        
        # Evaluate ensemble on test set
        y_pred_proba = ensemble_pipeline.predict_proba(X_test)[:, 1]
        
        if self.threshold_optimization:
            # Find and use optimal threshold
            optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            self.optimal_threshold = optimal_threshold
        else:
            # Use default threshold (0.5)
            y_pred = ensemble_pipeline.predict(X_test)
            self.optimal_threshold = 0.5
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print("\nEnsemble Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Compare with XGBoost alone
        if hasattr(self, 'xgb_pipeline'):
            print("\nComparing with XGBoost alone:")
            xgb_pred = self.xgb_pipeline.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
            print(f"Ensemble Accuracy: {accuracy:.4f}")
            print(f"Improvement: {(accuracy - xgb_accuracy) * 100:.2f}%")
        
        # Store ensemble model and predictions
        self.ensemble_pipeline = ensemble_pipeline
        self.ensemble_pred = y_pred
        self.ensemble_pred_proba = y_pred_proba
        
        return ensemble_pipeline, accuracy
    
    def shap_analysis(self, n_samples=500):
        """
        Perform SHAP (SHapley Additive exPlanations) analysis for model interpretability
        
        Args:
            n_samples: Number of samples to use for SHAP analysis
            
        Returns:
            SHAP values
        """
        if not SHAP_AVAILABLE:
            print("Error: SHAP is not installed. Please install with 'pip install shap'")
            return None
        
        # Ensure model is trained
        if not hasattr(self, 'xgb_pipeline'):
            print("Error: Model not trained. Please train model first.")
            return None
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        print("\n=== SHAP Analysis ===")
        
        # Get preprocessed data
        imputer = self.xgb_pipeline.named_steps['imputer']
        scaler = self.xgb_pipeline.named_steps['scaler']
        X_test_processed = scaler.transform(imputer.transform(X_test))
        
        # Sample for speed if needed
        if n_samples < len(X_test_processed):
            sample_indices = np.random.choice(len(X_test_processed), size=n_samples, replace=False)
            X_sample = X_test_processed[sample_indices]
            sample_original_indices = sample_indices  # To map back to original test indices
        else:
            X_sample = X_test_processed
            sample_original_indices = np.arange(len(X_test_processed))
        
        # Get classifier from pipeline
        clf = self.xgb_pipeline.named_steps['classifier']
        
        # Create explainer
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(clf)
        
        # Calculate SHAP values
        print(f"Calculating SHAP values for {len(X_sample)} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # SHAP summary plot
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'))
        plt.close()
        print(f"SHAP summary plot saved to {os.path.join(self.output_dir, 'shap_summary.png')}")
        
        # SHAP dependence plots for top features
        top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]  # Top 5 features
        
        print("Generating SHAP dependence plots for top features...")
        for feature in top_features:
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(feature, shap_values, X_sample, 
                               feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'shap_dependence_{self.feature_names[feature]}.png'))
            plt.close()
        
        print(f"SHAP dependence plots saved to {self.output_dir}")
        
        # Analyze misclassified examples using SHAP
        if hasattr(self, 'y_pred'):
            misclassified = y_test[sample_original_indices] != self.y_pred[sample_original_indices]
            if np.sum(misclassified) > 0:
                print("\nAnalyzing misclassified examples with SHAP...")
                plt.figure(figsize=(15, 10))
                shap.decision_plot(explainer.expected_value, shap_values[misclassified], 
                                 feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'shap_misclassified.png'))
                plt.close()
                print(f"SHAP decision plot for misclassified examples saved to {os.path.join(self.output_dir, 'shap_misclassified.png')}")
        
        # Store SHAP results
        self.shap_values = shap_values
        self.shap_explainer = explainer
        
        return shap_values
    
    def save_model(self, model_path=None):
        """
        Save the trained model to disk
        
        Args:
            model_path: Path to save the model
        """
        if not hasattr(self, 'xgb_pipeline'):
            print("Error: Model not trained. Please train model first.")
            return
        
        # Default path if not specified
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'dolphin_classifier_model.pkl')
        
        print(f"\nSaving model to {model_path}")
        
        # Create model info
        model_info = {
            'model': self.xgb_pipeline,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5),
            'feature_names': self.feature_names,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'accuracy': accuracy_score(self.y_test, self.y_pred),
                'auc_roc': roc_auc_score(self.y_test, self.y_pred_proba),
                'f1': f1_score(self.y_test, self.y_pred)
            }
        }
        
        # Add ensemble model if available
        if hasattr(self, 'ensemble_pipeline'):
            model_info['ensemble_model'] = self.ensemble_pipeline
            model_info['metrics']['ensemble_accuracy'] = accuracy_score(
                self.y_test, self.ensemble_pred)
        
        # Add feature selector if available
        if hasattr(self, 'feature_selection_results') and 'selector' in self.feature_selection_results:
            model_info['feature_selector'] = self.feature_selection_results['selector']
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print("Model saved successfully!")
        
    def load_model(self, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        print(f"\nLoading model from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            # Load model and related info
            self.xgb_pipeline = model_info['model']
            self.optimal_threshold = model_info.get('optimal_threshold', 0.5)
            self.feature_names = model_info.get('feature_names', self.feature_names)
            
            if 'ensemble_model' in model_info:
                self.ensemble_pipeline = model_info['ensemble_model']
            
            if 'feature_selector' in model_info:
                self.feature_selector = model_info['feature_selector']
            
            # Print model info
            print("Model loaded successfully!")
            print("Model information:")
            print(f"  Training date: {model_info.get('training_date', 'Unknown')}")
            if 'metrics' in model_info:
                print("  Performance metrics:")
                for metric, value in model_info['metrics'].items():
                    print(f"    {metric}: {value:.4f}")
            
            return self.xgb_pipeline
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def generate_report(self):
        """
        Generate a comprehensive report of model performance and analysis
        """
        if not hasattr(self, 'xgb_pipeline'):
            print("Error: Model not trained. Please train model first.")
            return
        
        print("\n=== Generating Comprehensive Report ===")
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Open report file
        report_path = os.path.join(report_dir, 'model_report.html')
        with open(report_path, 'w') as f:
            # Write header
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dolphin Whistle Classifier Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }
                    h1, h2, h3 { color: #2c3e50; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .section { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                    .highlight { background: #e7f4ff; padding: 10px; border-left: 4px solid #3498db; }
                    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                    th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    .metric-good { color: green; }
                    .metric-bad { color: red; }
                    .metric-neutral { color: orange; }
                    img { max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Dolphin Whistle Classifier Report</h1>
                    <p>Generated on: """+pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')+"""</p>
            """)
            
            # Dataset Section
            f.write("""
                <div class="section">
                    <h2>1. Dataset Information</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total samples</td><td>"""+str(self.n_samples)+"""</td></tr>
                        <tr><td>Positive samples (Whistles)</td><td>"""+str(np.sum(self.truth == 1))+"""</td></tr>
                        <tr><td>Negative samples (Noise)</td><td>"""+str(np.sum(self.truth == 0))+"""</td></tr>
                        <tr><td>Class balance</td><td>"""+f"{np.sum(self.truth == 1)/self.n_samples:.2%} positive"+"""</td></tr>
                        <tr><td>Number of features</td><td>"""+str(self.n_features)+"""</td></tr>
                        <tr><td>Time metrics</td><td>"""+("Excluded" if self.skip_time_metrics else "Included")+"""</td></tr>
                    </table>
                </div>
            """)
            
            # Model Performance
            if hasattr(self, 'y_pred') and hasattr(self, 'y_test'):
                accuracy = accuracy_score(self.y_test, self.y_pred)
                auc_roc = roc_auc_score(self.y_test, self.y_pred_proba)
                f1 = f1_score(self.y_test, self.y_pred)
                precision = precision_score(self.y_test, self.y_pred)
                recall = recall_score(self.y_test, self.y_pred)
                
                f.write("""
                    <div class="section">
                        <h2>2. Model Performance</h2>
                        <table>
                            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>"""+f"{accuracy:.4f}"+"""</td>
                                <td class=\""""+("metric-good" if accuracy > 0.85 else "metric-neutral" if accuracy > 0.75 else "metric-bad")+"""\">
                                """+("Excellent" if accuracy > 0.85 else "Good" if accuracy > 0.75 else "Needs Improvement")+"""
                                </td>
                            </tr>
                            <tr>
                                <td>AUC-ROC</td>
                                <td>"""+f"{auc_roc:.4f}"+"""</td>
                                <td class=\""""+("metric-good" if auc_roc > 0.85 else "metric-neutral" if auc_roc > 0.75 else "metric-bad")+"""\">
                                """+("Excellent" if auc_roc > 0.85 else "Good" if auc_roc > 0.75 else "Needs Improvement")+"""
                                </td>
                            </tr>
                            <tr>
                                <td>F1 Score</td>
                                <td>"""+f"{f1:.4f}"+"""</td>
                                <td class=\""""+("metric-good" if f1 > 0.85 else "metric-neutral" if f1 > 0.75 else "metric-bad")+"""\">
                                """+("Excellent" if f1 > 0.85 else "Good" if f1 > 0.75 else "Needs Improvement")+"""
                                </td>
                            </tr>
                            <tr><td>Precision</td><td>"""+f"{precision:.4f}"+"""</td><td>Ratio of true positive predictions to all positive predictions</td></tr>
                            <tr><td>Recall</td><td>"""+f"{recall:.4f}"+"""</td><td>Ratio of true positive predictions to all actual positives</td></tr>
                            <tr><td>Classification Threshold</td><td>"""+f"{self.optimal_threshold:.4f}"+"""</td><td>"""+("Optimized" if self.threshold_optimization else "Default")+"""</td></tr>
                        </table>
                        
                        <h3>Confusion Matrix</h3>
                        <p>The confusion matrix shows the counts of true positives, false positives, true negatives, and false negatives:</p>
                        <img src="../confusion_matrix.png" alt="Confusion Matrix">
                        
                        <h3>ROC Curve</h3>
                        <p>The ROC curve shows the trade-off between true positive rate and false positive rate:</p>
                        <img src="../roc_curve.png" alt="ROC Curve">
                    </div>
                """)
            
            # Model Parameters
            if hasattr(self, 'clf'):
                f.write("""
                    <div class="section">
                        <h2>3. Model Configuration</h2>
                        <h3>XGBoost Parameters</h3>
                        <table>
                            <tr><th>Parameter</th><th>Value</th></tr>
                """)
                
                # Get parameters
                params = self.clf.get_params()
                for param, value in sorted(params.items()):
                    f.write(f"<tr><td>{param}</td><td>{value}</td></tr>")
                
                f.write("""
                        </table>
                    </div>
                """)
            
            # Feature Importance
            if hasattr(self, 'clf') and hasattr(self.clf, 'feature_importances_'):
                f.write("""
                    <div class="section">
                        <h2>4. Feature Importance</h2>
                        <p>The top features ranked by importance:</p>
                        <img src="../feature_importances.png" alt="Feature Importance">
                        
                        <h3>Top 10 Most Important Features</h3>
                        <table>
                            <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
                """)
                
                importances = self.clf.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                for i in range(min(10, len(indices))):
                    f.write(f"<tr><td>{i+1}</td><td>{self.feature_names[indices[i]]}</td><td>{importances[indices[i]]:.4f}</td></tr>")
                
                f.write("""
                        </table>
                    </div>
                """)
            
            # Analysis of Overfitting
            if hasattr(self, 'learning_curve_results'):
                f.write("""
                    <div class="section">
                        <h2>5. Bias-Variance Analysis</h2>
                        <p>Learning curves show the model's performance as the training set size increases:</p>
                        <img src="../learning_curves.png" alt="Learning Curves">
                        
                        <div class="highlight">
                """)
                
                # Calculate gap between training and CV performance
                train_mean = self.learning_curve_results['train_mean']
                test_mean = self.learning_curve_results['test_mean']
                gap = train_mean[-1] - test_mean[-1]
                
                if gap > 0.1:
                    f.write("""
                        <h3>Diagnosis: High Variance (Overfitting)</h3>
                        <p>The model performs significantly better on the training data than on validation data.</p>
                        <h4>Recommendations:</h4>
                        <ul>
                            <li>Increase regularization (higher alpha/lambda)</li>
                            <li>Reduce model complexity (lower max_depth)</li>
                            <li>Add more training data</li>
                            <li>Feature selection/reduction</li>
                        </ul>
                    """)
                elif gap < 0.05 and train_mean[-1] < 0.85:
                    f.write("""
                        <h3>Diagnosis: High Bias (Underfitting)</h3>
                        <p>The model has limited performance on both training and validation data.</p>
                        <h4>Recommendations:</h4>
                        <ul>
                            <li>Increase model complexity (higher max_depth)</li>
                            <li>Reduce regularization</li>
                            <li>Add more features or feature interactions</li>
                            <li>Consider more powerful models</li>
                        </ul>
                    """)
                else:
                    f.write("""
                        <h3>Diagnosis: Good Balance between Bias and Variance</h3>
                        <p>The model has good performance with a reasonable gap between training and validation scores.</p>
                    """)
                
                f.write("""
                        </div>
                    </div>
                """)
            
            # Feature Selection Results
            if hasattr(self, 'feature_selection_results'):
                f.write("""
                    <div class="section">
                        <h2>6. Feature Selection</h2>
                """)
                
                if 'selected_indices' in self.feature_selection_results:
                    n_selected = len(self.feature_selection_results['selected_indices'])
                    f.write(f"""
                        <p>Selected {n_selected} out of {self.n_features} features</p>
                        <h3>Performance Comparison</h3>
                        <table>
                            <tr><th>Metric</th><th>Full Feature Set</th><th>Selected Features</th><th>Difference</th></tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>{self.feature_selection_results['accuracy_full']:.4f}</td>
                                <td>{self.feature_selection_results['accuracy_selected']:.4f}</td>
                                <td class=\"{("metric-good" if self.feature_selection_results['accuracy_selected'] >= self.feature_selection_results['accuracy_full'] else "metric-bad")}\">
                                    {(self.feature_selection_results['accuracy_selected'] - self.feature_selection_results['accuracy_full']):.4f}
                                </td>
                            </tr>
                            <tr>
                                <td>AUC-ROC</td>
                                <td>{self.feature_selection_results['auc_full']:.4f}</td>
                                <td>{self.feature_selection_results['auc_selected']:.4f}</td>
                                <td class=\"{("metric-good" if self.feature_selection_results['auc_selected'] >= self.feature_selection_results['auc_full'] else "metric-bad")}\">
                                    {(self.feature_selection_results['auc_selected'] - self.feature_selection_results['auc_full']):.4f}
                                </td>
                            </tr>
                        </table>
                        
                        <h3>Top Selected Features</h3>
                        <table>
                            <tr><th>Feature</th></tr>
                    """)
                    
                    selected_indices = self.feature_selection_results['selected_indices']
                    for idx in selected_indices[:min(20, len(selected_indices))]:
                        f.write(f"<tr><td>{self.feature_names[idx]}</td></tr>")
                    
                    f.write("""
                        </table>
                    """)
                
                f.write("""
                    </div>
                """)
            
            # Ensemble Model Results
            if hasattr(self, 'ensemble_pred'):
                f.write("""
                    <div class="section">
                        <h2>7. Ensemble Model</h2>
                """)
                
                # Compare with XGBoost alone
                if hasattr(self, 'y_pred'):
                    xgb_accuracy = accuracy_score(self.y_test, self.y_pred)
                    ensemble_accuracy = accuracy_score(self.y_test, self.ensemble_pred)
                    xgb_auc = roc_auc_score(self.y_test, self.y_pred_proba)
                    ensemble_auc = roc_auc_score(self.y_test, self.ensemble_pred_proba)
                    
                    f.write(f"""
                        <h3>Performance Comparison</h3>
                        <table>
                            <tr><th>Metric</th><th>XGBoost</th><th>Ensemble</th><th>Improvement</th></tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>{xgb_accuracy:.4f}</td>
                                <td>{ensemble_accuracy:.4f}</td>
                                <td class=\"{("metric-good" if ensemble_accuracy > xgb_accuracy else "metric-bad")}\">
                                    {(ensemble_accuracy - xgb_accuracy):.4f} ({(ensemble_accuracy - xgb_accuracy) * 100:.2f}%)
                                </td>
                            </tr>
                            <tr>
                                <td>AUC-ROC</td>
                                <td>{xgb_auc:.4f}</td>
                                <td>{ensemble_auc:.4f}</td>
                                <td class=\"{("metric-good" if ensemble_auc > xgb_auc else "metric-bad")}\">
                                    {(ensemble_auc - xgb_auc):.4f} ({(ensemble_auc - xgb_auc) * 100:.2f}%)
                                </td>
                            </tr>
                        </table>
                        
                        <div class="highlight">
                            <p>The ensemble model combines XGBoost, Random Forest, and Logistic Regression for improved robustness.</p>
                            <p>Verdict: {("Ensemble provides better performance and should be used." if ensemble_accuracy > xgb_accuracy else "XGBoost alone provides similar or better performance.")}
                        </div>
                    """)
                
                f.write("""
                    </div>
                """)
            
            # SHAP Analysis
            if hasattr(self, 'shap_values'):
                f.write("""
                    <div class="section">
                        <h2>8. SHAP Analysis</h2>
                        <p>SHAP values show how each feature contributes to model predictions:</p>
                        <img src="../shap_summary.png" alt="SHAP Summary">
                        
                        <h3>Feature Interactions and Dependencies</h3>
                        <p>SHAP dependence plots show how feature values affect predictions:</p>
                """)
                
                # Find top feature dependence plots
                top_features = np.argsort(np.abs(self.shap_values).mean(0))[-5:]
                for feature in top_features:
                    feature_name = self.feature_names[feature]
                    f.write(f"""
                        <h4>Feature: {feature_name}</h4>
                        <img src="../shap_dependence_{feature_name}.png" alt="SHAP Dependence {feature_name}">
                    """)
                
                f.write("""
                    </div>
                """)
            
            # Conclusions and Recommendations
            f.write("""
                <div class="section">
                    <h2>9. Conclusions and Recommendations</h2>
                    <div class="highlight">
            """)
            
            # Generate automatic conclusions based on available data
            if hasattr(self, 'y_pred') and hasattr(self, 'y_test'):
                accuracy = accuracy_score(self.y_test, self.y_pred)
                # Model quality assessment
                if accuracy > 0.9:
                    f.write("<h3>Overall Model Quality: Excellent</h3>")
                elif accuracy > 0.8:
                    f.write("<h3>Overall Model Quality: Good</h3>")
                elif accuracy > 0.7:
                    f.write("<h3>Overall Model Quality: Reasonable</h3>")
                else:
                    f.write("<h3>Overall Model Quality: Needs Improvement</h3>")
            
            f.write("""
                        <h3>Key Findings:</h3>
                        <ul>
            """)
            
            # Add findings based on analyses performed
            if hasattr(self, 'learning_curve_results'):
                train_mean = self.learning_curve_results['train_mean']
                test_mean = self.learning_curve_results['test_mean']
                gap = train_mean[-1] - test_mean[-1]
                
                if gap > 0.1:
                    f.write("<li>The model shows signs of overfitting (high variance)</li>")
                elif gap < 0.05 and train_mean[-1] < 0.85:
                    f.write("<li>The model shows signs of underfitting (high bias)</li>")
                else:
                    f.write("<li>The model has a good balance between bias and variance</li>")
            
            if hasattr(self, 'feature_selection_results') and 'selected_indices' in self.feature_selection_results:
                n_selected = len(self.feature_selection_results['selected_indices'])
                perf_diff = self.feature_selection_results['accuracy_selected'] - self.feature_selection_results['accuracy_full']
                
                if perf_diff >= 0:
                    f.write(f"<li>Feature selection improved performance while reducing dimensionality from {self.n_features} to {n_selected} features</li>")
                else:
                    f.write(f"<li>Feature selection reduced dimensionality from {self.n_features} to {n_selected} features with a small performance trade-off ({perf_diff:.4f})</li>")
            
            if hasattr(self, 'ensemble_pred') and hasattr(self, 'y_pred'):
                ensemble_accuracy = accuracy_score(self.y_test, self.ensemble_pred)
                xgb_accuracy = accuracy_score(self.y_test, self.y_pred)
                
                if ensemble_accuracy > xgb_accuracy:
                    f.write(f"<li>Ensemble modeling improved accuracy by {(ensemble_accuracy - xgb_accuracy) * 100:.2f}%</li>")
                else:
                    f.write("<li>XGBoost alone provides optimal performance</li>")
            
            if hasattr(self, 'clf') and hasattr(self.clf, 'feature_importances_'):
                importances = self.clf.feature_importances_
                indices = np.argsort(importances)[::-1]
                top_feature = self.feature_names[indices[0]]
                f.write(f"<li>The most important feature is '{top_feature}'</li>")
            
            f.write("""
                        </ul>
                        
                        <h3>Recommendations:</h3>
                        <ul>
            """)
            
            # Add recommendations based on analyses
            if hasattr(self, 'learning_curve_results'):
                train_mean = self.learning_curve_results['train_mean']
                test_mean = self.learning_curve_results['test_mean']
                gap = train_mean[-1] - test_mean[-1]
                
                if gap > 0.1:
                    f.write("<li>Increase regularization to reduce overfitting</li>")
                    f.write("<li>Consider reducing model complexity (e.g., lower max_depth)</li>")
                    f.write("<li>Collect more training data if possible</li>")
                elif gap < 0.05 and train_mean[-1] < 0.85:
                    f.write("<li>Increase model complexity to address underfitting</li>")
                    f.write("<li>Consider adding more relevant features</li>")
                    f.write("<li>Explore more powerful modeling approaches</li>")
            
            if hasattr(self, 'optimal_threshold') and self.optimal_threshold != 0.5:
                f.write(f"<li>Use the optimized classification threshold of {self.optimal_threshold:.4f} for predictions</li>")
            
            # Add general recommendations
            f.write("<li>Monitor model performance over time as data distribution might change</li>")
            f.write("<li>Consider periodic retraining to maintain optimal performance</li>")
            
            f.write("""
                        </ul>
                    </div>
                </div>
            """)
            
            # Close HTML
            f.write("""
                </div>
            </body>
            </html>
            """)
        
        print(f"Comprehensive report generated at {report_path}")
        return report_path
    
    def save_misclassified_to_csv(self, misclassified_indices=None):
        """Save misclassified examples to CSV for further analysis
        
        Args:
            misclassified_indices: Optional list of indices to save. If None,
                                finds misclassified examples from test set
        """
        if not hasattr(self, 'y_pred') or not hasattr(self, 'y_test'):
            print("Error: No predictions available. Train the model first.")
            return
            
        if misclassified_indices is None:
            # Get misclassified examples
            misclassified_indices = np.where(self.y_pred != self.y_test)[0]
        
        # Check if any misclassified examples exist
        if len(misclassified_indices) == 0:
            print("No misclassified examples found.")
            return
        
        # Create base DataFrame with truth and prediction
        base_data = {
            'true_label': self.y_test[misclassified_indices],
            'predicted_label': self.y_pred[misclassified_indices],
            'prediction_probability': self.y_pred_proba[misclassified_indices],
            'index': self.index[misclassified_indices] if hasattr(self, 'index') else misclassified_indices
        }
        
        # Get feature data
        feature_data = {}
        for i, name in enumerate(self.feature_names):
            if i < self.X_test.shape[1]:
                feature_data[name] = self.X_test[misclassified_indices, i]
        
        # Combine base data and feature data
        misclassified_data = pd.DataFrame(base_data)
        feature_df = pd.DataFrame(feature_data)
        misclassified_data = pd.concat([misclassified_data, feature_df], axis=1)
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, 'misclassified_examples.csv')
        misclassified_data.to_csv(output_file, index=False)
        print(f"\nMisclassified examples saved to {output_file}")
        return misclassified_indices

    def analyze_misclassifications(self):
        """
        Analyze misclassified examples with detailed visualizations
        """
        if not hasattr(self, 'y_pred') or not hasattr(self, 'y_test'):
            print("Error: No predictions available. Train the model first.")
            return
        
        # Find misclassified examples
        misclassified_indices = np.where(self.y_pred != self.y_test)[0]
        num_misclassified = len(misclassified_indices)
        
        if num_misclassified == 0:
            print("No misclassified examples found.")
            return
        
        print(f"\n=== Misclassification Analysis ===")
        print(f"Total misclassified examples: {num_misclassified} out of {len(self.y_test)} ({num_misclassified/len(self.y_test)*100:.2f}%)")
        
        # Count false positives and false negatives
        false_positives = np.sum((self.y_pred == 1) & (self.y_test == 0))
        false_negatives = np.sum((self.y_pred == 0) & (self.y_test == 1))
        
        # Calculate AUC-ROC for each class
        auc_roc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print(f"\nMetrics:")
        print(f"False positives (noise classified as whistle): {false_positives}")
        print(f"False negatives (whistle classified as noise): {false_negatives}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        
        # Get prediction probabilities for misclassified examples
        misclassified_probs = self.y_pred_proba[misclassified_indices]
        
        # Plot histogram of prediction probabilities for misclassified examples
        plt.figure(figsize=(10, 6))
        plt.hist(misclassified_probs, bins=20, alpha=0.7)
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold')
        if hasattr(self, 'optimal_threshold'):
            plt.axvline(x=self.optimal_threshold, color='g', linestyle='--', 
                      label=f'Optimal threshold ({self.optimal_threshold:.3f})')
        plt.title('Prediction Probabilities for Misclassified Examples')
        plt.xlabel('Probability of Whistle')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'misclassified_probabilities.png'))
        plt.close()
        
        print(f"Misclassified probabilities plot saved to {os.path.join(self.output_dir, 'misclassified_probabilities.png')}")
        
        # Analyze feature values for misclassified examples
        misclassified_features = self.X_test[misclassified_indices]
        
        # Get feature importance
        if hasattr(self.clf, 'feature_importances_'):
            importances = self.clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Get top 5 important features
            top_features_idx = indices[:5]
            top_features_names = [self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}' 
                                for i in top_features_idx]
            
            # Compare feature values for misclassified vs correctly classified
            correctly_classified_indices = np.where(self.y_pred == self.y_test)[0]
            correctly_classified_features = self.X_test[correctly_classified_indices]
            
            # Plot feature distributions for top features
            plt.figure(figsize=(15, 10))
            
            for i, (idx, name) in enumerate(zip(top_features_idx, top_features_names)):
                plt.subplot(2, 3, i+1)
                
                # Plot distributions
                plt.hist(correctly_classified_features[:, idx], bins=20, alpha=0.5, label='Correct')
                plt.hist(misclassified_features[:, idx], bins=20, alpha=0.5, label='Misclassified')
                
                plt.title(f'Feature: {name}')
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'misclassified_feature_distributions.png'))
            plt.close()
            
            print(f"Misclassified feature distributions plot saved to {os.path.join(self.output_dir, 'misclassified_feature_distributions.png')}")
        
        # Analyze misclassifications by true class
        fp_indices = np.where((self.y_pred == 1) & (self.y_test == 0))[0]
        fn_indices = np.where((self.y_pred == 0) & (self.y_test == 1))[0]
        
        # Plot confusion matrix with percentages
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=['Noise', 'Whistle'], 
                   yticklabels=['Noise', 'Whistle'])
        plt.title('Confusion Matrix (Percentage)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix_percent.png'))
        plt.close()
        
        print(f"Percentage confusion matrix saved to {os.path.join(self.output_dir, 'confusion_matrix_percent.png')}")
        
        # Save misclassified examples to CSV
        self.save_misclassified_to_csv(misclassified_indices)
        
        # Return misclassified indices for further analysis
        return misclassified_indices, fp_indices, fn_indices


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Dolphin Whistle Classifier')

    parser.add_argument('--analyze_misclassifications', action='store_true',
                  help='Perform detailed misclassification analysis')
    
    # Input/output options
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to the metrics CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (default: same as metrics file)')
    
    # Testing and debug options
    parser.add_argument('--test', action='store_true',
                        help='Run in small test mode (subset of data)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Feature handling options
    parser.add_argument('--include_time_metrics', action='store_true',
                        help='Include time metrics in feature set')
    
    # Training options
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Optimize classification threshold')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping during training')
    
    # Tuning options
    parser.add_argument('--tune_bayesian', action='store_true',
                        help='Perform Bayesian parameter tuning with Optuna')
    parser.add_argument('--tune_trials', type=int, default=100,
                        help='Number of trials for Bayesian optimization')
    
    # Analysis options
    parser.add_argument('--nested_cv', action='store_true',
                        help='Perform nested cross-validation')
    parser.add_argument('--learning_curves', action='store_true',
                        help='Generate learning curves')
    parser.add_argument('--validation_curves', action='store_true',
                        help='Generate validation curves')
    parser.add_argument('--feature_selection', action='store_true',
                        help='Perform feature selection')
    parser.add_argument('--ensemble', action='store_true',
                        help='Create ensemble of models')
    parser.add_argument('--shap', action='store_true',
                        help='Perform SHAP analysis')
    
    # Output options
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model to disk')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Load model from disk instead of training')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate comprehensive HTML report')
    
    # Run all analyses
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses (except tuning)')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.metrics_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create enhanced classifier
    classifier = EnhancedDolphinClassifier(
        metrics_file=args.metrics_file,
        random_state=args.random_state,
        threshold_optimization=args.optimize_threshold,
        skip_time_metrics=not args.include_time_metrics
    )
    
    # If loading model instead of training
    if args.load_model:
        classifier.load_model(args.load_model)
    else:
        # Train the model
        if args.tune_bayesian:
            # Tune with Bayesian optimization
            if not OPTUNA_AVAILABLE:
                print("Warning: Optuna not installed. Falling back to standard training.")
                classifier.train_with_early_stopping(early_stopping=args.early_stopping)
            else:
                classifier.tune_with_optuna(n_trials=args.tune_trials, small_test=args.test)
        else:
            # Train with/without early stopping
            classifier.train_with_early_stopping(early_stopping=args.early_stopping)
    
    # Run analyses based on arguments
    
    # Nested cross-validation
    if args.nested_cv or args.all:
        classifier.nested_cross_validation(
            outer_cv=3 if args.test else 5,
            inner_cv=2 if args.test else 3,
            small_test=args.test
        )
    
    # Learning curves
    if args.learning_curves or args.all:
        classifier.plot_learning_curves(
            train_sizes=np.linspace(0.1, 1.0, 5 if args.test else 10),
            cv=3 if args.test else 5
        )
    
    # Validation curves
    if args.validation_curves or args.all:
        if args.test:
            # Simplified parameter list for test mode
            param_list = [
                ('max_depth', [3, 5, 7]),
                ('learning_rate', [0.01, 0.05, 0.1]),
                ('n_estimators', [50, 100, 200])
            ]
            classifier.plot_validation_curves(param_list=param_list)
        else:
            classifier.plot_validation_curves()

        # Feature selection
    if args.feature_selection or args.all:
        use_shap = args.shap or (SHAP_AVAILABLE and args.all)
        classifier.select_features(threshold=0.90, use_shap=use_shap)
    
    # Create ensemble model
    if args.ensemble or args.all:
        classifier.create_model_ensemble()
    
    # SHAP analysis
    if args.shap or args.all:
        if not SHAP_AVAILABLE:
            print("Warning: SHAP not installed. Skipping SHAP analysis.")
        else:
            classifier.shap_analysis(n_samples=100 if args.test else 500)
    
    # Save model to disk
    if args.save_model or args.all:
        classifier.save_model()
    
    # Generate comprehensive report
    if args.generate_report or args.all:
        classifier.generate_report()

    if args.analyze_misclassifications or args.all:
        classifier.analyze_misclassifications()
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    # Check for required packages
    missing_packages = []
    try:
        import xgboost
    except ImportError:
        missing_packages.append("xgboost")
    
    try:
        import optuna
        OPTUNA_AVAILABLE = True
    except ImportError:
        missing_packages.append("optuna (optional)")
    
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        missing_packages.append("shap (optional)")
    
    if missing_packages:
        print("Warning: Some required packages are missing:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("Install with: pip install [package_name]")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()




# # Train a basic model with default settings
# python enhanced_dolphin_classifier.py --metrics_file your_metrics.csv

# # Run all analyses (learning curves, validation curves, feature selection, ensemble, report)
# python enhanced_dolphin_classifier.py --metrics_file your_metrics.csv --all

# # Perform Bayesian optimization with 100 trials
# python enhanced_dolphin_classifier.py --metrics_file your_metrics.csv --tune_bayesian --tune_trials 100

# # Quick test mode with a small subset of data
# python enhanced_dolphin_classifier.py --metrics_file your_metrics.csv --test --all
