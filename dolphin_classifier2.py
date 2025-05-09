import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve, accuracy_score, roc_auc_score
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import joblib

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from dolphin_detector import config
except ImportError:
    import config

class DolphinClassifier:
    """
    Classifier for dolphin whistle detection using XGBoost
    """
    def __init__(self, metrics_file, random_state=config.RANDOM_STATE, threshold_optimization=False, skip_time_metrics=True):
        """
        Initialize the classifier
        
        Args:
            metrics_file: Path to the metrics CSV file
            random_state: Random seed for reproducibility
            threshold_optimization: Whether to use threshold optimization (default: False)
            skip_time_metrics: Whether to skip time metrics in feature extraction (default: True)
        """
        self.metrics_file = metrics_file
        self.random_state = random_state
        self.threshold_optimization = threshold_optimization
        self.skip_time_metrics = skip_time_metrics
        self.load_data()
        self.scaler = StandardScaler()
        
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

        # Add new audio feature names
        self.feature_names.extend([
            'zero_crossing_rate',
            'spectral_centroid', 'spectral_centroid_std', 'spectral_centroid_skew',
            'spectral_bandwidth', 'spectral_bandwidth_std', 'spectral_bandwidth_skew',
            'spectral_rolloff', 'spectral_rolloff_std', 'spectral_rolloff_skew',
            'spectral_contrast', 'spectral_contrast_std', 'spectral_contrast_skew',
            'chroma_mean', 'chroma_std',
            'energy'
        ])

        # Add MFCC feature names
        for i in range(1, 21):  # 20 MFCCs
            self.feature_names.extend([
                f'mfcc_{i}_mean',
                f'mfcc_{i}_std',
                f'mfcc_{i}_skew',
                f'mfcc_{i}_max',
                f'mfcc_{i}_min'
            ])
        
        # Print dataset statistics
        self.n_samples, self.n_features = self.features.shape
        n_positive = np.sum(self.truth == 1)
        n_negative = np.sum(self.truth == 0)
        
        print(f"Loaded {self.n_samples} samples with {self.n_features} features")
        print("\nFeatures used:")
        print("\n1. Template Matching Metrics:")
        for enhancement in ['Vertical', 'Horizontal']:
            print(f"  {enhancement} Enhancement:")
            for prefix in ['max', 'xLoc', 'yLoc']:
                print(f"    - {prefix} scores for {n_templates} templates")
        
        if not self.skip_time_metrics:
            print("\n2. Time Metrics:")
            print("  - centTime: Centroid time metrics")
            print("  - bwTime: Bandwidth time metrics")
            print("  - skewTime: Skewness time metrics")
            print("  - tvTime: Time-varying metrics")
        
        print("\n3. High Frequency Metrics:")
        print("  - CentStd: Standard deviation of centroids")
        print("  - AvgBwd: Mean bandwidth")
        print("  - hfCent: High-frequency centroid")
        print("  - hfBwd: High-frequency bandwidth")
        
        print("\n4. High Frequency Template Matching:")
        print("  - hfMax: Bar template (18px wide)")
        print("  - hfMax2: Bar1 template (24px wide)")
        print("  - hfMax3: Bar2 template (12px wide)")

        print("\n5. Audio Signal Features:")
        print("  - Zero crossing rate")
        print("  - Spectral features (centroid, bandwidth, rolloff, contrast)")
        print("  - Chroma features")
        print("  - Energy")
        print("  - 20 MFCCs with statistics (mean, std, skew, max, min)")
        
        print(f"\nPositive samples (whistles): {n_positive}")
        print(f"Negative samples (noise): {n_negative}")
        print(f"Time metrics: {'excluded' if self.skip_time_metrics else 'included'}")
        
    def prepare_data(self, test_size=config.TEST_SIZE, small_test=False):
        """
        Prepare data for training and testing
        
        Args:
            test_size: Proportion of data to use for testing
            small_test: If True, use only a small subset of data for quick testing
            
        Returns:
            X_train, X_test, y_train, y_test: Training and testing data
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
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.truth, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=self.truth
        )
        
        return X_train, X_test, y_train, y_test
    
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

    def predict_with_threshold(self, X, threshold=0.5):
        """
        Make predictions using a specific threshold
        
        Args:
            X: Features to predict on
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            tuple: (Binary predictions, Prediction probabilities)
        """
        if not hasattr(self, 'xgb_pipeline'):
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Get prediction probabilities
        pred_proba = self.xgb_pipeline.predict_proba(X)[:, 1]
        
        # Apply threshold
        predictions = (pred_proba >= threshold).astype(int)
        
        return predictions, pred_proba

    def train_model(self):
        """Train the XGBoost classifier with optional threshold optimization"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Define XGBoost parameters
        xgb_params = {
            'max_depth': 5,
            'subsample': 0.8,
            'n_estimators': 100,
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': self.random_state
        }
        
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        
        # Create pipeline for the classifier
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb_clf)
        ])
        
        # Train XGBoost Classifier
        print("Training XGBoost Classifier...")
        xgb_pipeline.fit(X_train, y_train)
        
        # Get predictions
        xgb_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
        
        if self.threshold_optimization:
            # Find and use optimal threshold
            optimal_threshold = self.find_optimal_threshold(y_test, xgb_pred_proba)
            xgb_pred, _ = self.predict_with_threshold(X_test, threshold=optimal_threshold)
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
        
        return accuracy_score(y_test, xgb_pred)

    def cross_validate(self, n_folds=config.CV_FOLDS, small_test=False):
        """Perform cross-validation with optional threshold optimization"""
        # Use best parameters if available from tuning, otherwise use defaults
        if hasattr(self, 'best_xgb_params'):
            xgb_params = {k.replace('classifier__', ''): v for k, v in self.best_xgb_params.items()}
            print("Using best parameters from tuning:")
            for param, value in xgb_params.items():
                print(f"  {param}: {value}")
        else:
            xgb_params = config.XGB_PARAMS.copy()
            if small_test:
                xgb_params['n_estimators'] = 20
            print("Using default parameters (no tuning results available)")
        
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        
        # Create pipeline for the classifier
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb_clf)
        ])
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Arrays to store results
        xgb_scores = []
        xgb_auc_scores = []
        optimal_thresholds = [] if self.threshold_optimization else None
        
        print(f"Performing {n_folds}-fold cross-validation with XGBoost...")
        for i, (train_idx, test_idx) in enumerate(tqdm(cv.split(self.features, self.truth), total=n_folds, desc="Cross-validation")):
            X_train, X_test = self.features[train_idx], self.features[test_idx]
            y_train, y_test = self.truth[train_idx], self.truth[test_idx]
            
            # Train and evaluate XGBoost
            xgb_pipeline.fit(X_train, y_train)
            xgb_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
            
            if self.threshold_optimization:
                # Find optimal threshold for this fold
                optimal_threshold = self.find_optimal_threshold(y_test, xgb_pred_proba)
                optimal_thresholds.append(optimal_threshold)
                xgb_pred = (xgb_pred_proba >= optimal_threshold).astype(int)
            else:
                # Use default threshold
                xgb_pred = xgb_pipeline.predict(X_test)
            
            # Calculate metrics
            xgb_score = accuracy_score(y_test, xgb_pred)
            xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
            
            xgb_scores.append(xgb_score)
            xgb_auc_scores.append(xgb_auc)
            
            print(f"  Fold {i+1}:")
            print(f"    Accuracy={xgb_score:.4f}, AUC-ROC={xgb_auc:.4f}")
            if self.threshold_optimization:
                print(f"    Optimal threshold={optimal_threshold:.4f}")
        
        # Convert to numpy arrays
        xgb_scores = np.array(xgb_scores)
        xgb_auc_scores = np.array(xgb_auc_scores)
        if self.threshold_optimization:
            optimal_thresholds = np.array(optimal_thresholds)
        
        # Print summary
        print("\nCross-validation summary:")
        print(f"Accuracy: {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
        print(f"AUC-ROC:  {np.mean(xgb_auc_scores):.4f} ± {np.std(xgb_auc_scores):.4f}")
        if self.threshold_optimization:
            print(f"Optimal threshold: {np.mean(optimal_thresholds):.4f} ± {np.std(optimal_thresholds):.4f}")
        
        # Plot cross-validation results
        self.plot_xgboost_cv(xgb_scores, xgb_auc_scores, optimal_thresholds)
        
        # After calculating optimal thresholds, store the mean
        if self.threshold_optimization and optimal_thresholds is not None:
            self.optimal_threshold = np.mean(optimal_thresholds)
        
        return xgb_scores, xgb_auc_scores, optimal_thresholds

    def plot_xgboost_cv(self, xgb_scores, xgb_auc_scores, optimal_thresholds=None):
        """Plot cross-validation results including threshold distribution"""
        plt.figure(figsize=(15, 6))
        
        if optimal_thresholds is not None:
            # Create three subplots
            plt.subplot(1, 3, 1)
        
        # Create box plots for metrics
        bp = plt.boxplot([xgb_scores, xgb_auc_scores], 
                        labels=['Accuracy', 'AUC-ROC'], 
                        patch_artist=True)
        
        # Add individual points
        for i, scores in enumerate([xgb_scores, xgb_auc_scores], 1):
            x = np.random.normal(i, 0.04, size=len(scores))
            plt.scatter(x, scores, alpha=0.6)
        
        plt.title('Cross-Validation Metrics')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if optimal_thresholds is not None:
            # Plot threshold distribution
            plt.subplot(1, 3, 2)
            plt.hist(optimal_thresholds, bins=10, alpha=0.7)
            plt.axvline(x=np.mean(optimal_thresholds), color='r', linestyle='--',
                       label=f'Mean threshold: {np.mean(optimal_thresholds):.3f}')
            plt.title('Optimal Threshold Distribution')
            plt.xlabel('Threshold Value')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot relationship between threshold and accuracy
            plt.subplot(1, 3, 3)
            plt.scatter(optimal_thresholds, xgb_scores, alpha=0.6)
            plt.xlabel('Threshold Value')
            plt.ylabel('Accuracy')
            plt.title('Threshold vs Accuracy')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(self.metrics_file))
        plt.savefig(os.path.join(output_dir, 'xgboost_cv.png'))
        plt.close()
        
        print(f"Cross-validation plot saved to {os.path.join(output_dir, 'xgboost_cv.png')}")
    
    def plot_roc_curve(self):
        """
        Plot the ROC curve for the classifier
        """
        if not hasattr(self, 'y_pred_proba'):
            print("No predictions available. Train the classifier first.")
            return
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=config.FIGURE_SIZE)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add threshold annotations at key points
        threshold_points = [0.2, 0.4, 0.6, 0.8]
        for threshold in threshold_points:
            idx = (np.abs(thresholds - threshold)).argmin()
            plt.annotate(f'threshold={threshold:.1f}',
                        xy=(fpr[idx], tpr[idx]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(self.metrics_file))
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=config.DPI)
        plt.close()
        
        print(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")
        
    def plot_precision_recall_curve(self):
        """
        Plot the precision-recall curve for the classifier
        """
        if not hasattr(self, 'y_pred_proba'):
            print("No predictions available. Train the classifier first.")
            return
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'Precision-Recall curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(self.metrics_file))
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
        
        print(f"Precision-Recall curve saved to {os.path.join(output_dir, 'precision_recall_curve.png')}")
        
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the classifier
        """
        if not hasattr(self, 'y_pred'):
            print("No predictions available. Train the classifier first.")
            return
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Noise', 'Whistle'], 
                    yticklabels=['Noise', 'Whistle'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(self.metrics_file))
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
        
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
        """
        if not hasattr(self, 'clf'):
            print("No classifier available. Train the classifier first.")
            return
        
        # Get feature importance
        if hasattr(self.clf, 'feature_importances_'):
            # For tree-based models
            importances = self.clf.feature_importances_
            
            # Handle different estimator structures
            if hasattr(self.clf, 'estimators_') and isinstance(self.clf.estimators_, list):
                # For RandomForest
                std = np.std([tree.feature_importances_ for tree in self.clf.estimators_], axis=0)
            elif hasattr(self.clf, 'estimators_') and isinstance(self.clf.estimators_, np.ndarray):
                # For GradientBoosting
                std = np.zeros_like(importances)
            else:
                std = np.zeros_like(importances)
            
            indices = np.argsort(importances)[::-1]
            
            # Filter indices to only include those with valid feature names
            valid_indices = [idx for idx in indices if idx < len(self.feature_names)]
            valid_features = min(top_n, len(valid_indices))
            
            if valid_features == 0:
                print("\nNo valid features found with importance scores.")
                return
            
            # Print top features
            print(f"\nTop {valid_features} features:")
            for i in range(valid_features):
                idx = valid_indices[i]
                print(f"{i+1}. {self.feature_names[idx]} ({importances[idx]:.4f})")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance")
            
            # Use only valid indices for plotting
            plot_importances = [importances[idx] for idx in valid_indices[:valid_features]]
            plot_std = [std[idx] for idx in valid_indices[:valid_features]]
            plot_names = [self.feature_names[idx] for idx in valid_indices[:valid_features]]
            
            plt.bar(range(valid_features), plot_importances,
                    color="r", yerr=plot_std, align="center")
            plt.xticks(range(valid_features), plot_names, rotation=90)
            plt.xlim([-1, valid_features])
            plt.tight_layout()
            
            # Save the plot
            output_dir = os.path.join(os.path.dirname(self.metrics_file))
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()
            
            print(f"\nFeature importance plot saved to {os.path.join(output_dir, 'feature_importance.png')}")
        else:
            # For models without feature_importances_ attribute, use permutation importance
            result = permutation_importance(self.clf, self.X_test, self.y_test, 
                                           n_repeats=10, random_state=self.random_state)
            importances = result.importances_mean
            std = result.importances_std
            indices = np.argsort(importances)[::-1]
            
            # Filter indices to only include those with valid feature names
            valid_indices = [idx for idx in indices if idx < len(self.feature_names)]
            valid_features = min(top_n, len(valid_indices))
            
            if valid_features == 0:
                print("\nNo valid features found with importance scores.")
                return
            
            # Print top features
            print(f"\nTop {valid_features} features (permutation importance):")
            for i in range(valid_features):
                idx = valid_indices[i]
                print(f"{i+1}. {self.feature_names[idx]} ({importances[idx]:.4f})")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance (Permutation)")
            
            # Use only valid indices for plotting
            plot_importances = [importances[idx] for idx in valid_indices[:valid_features]]
            plot_std = [std[idx] for idx in valid_indices[:valid_features]]
            plot_names = [self.feature_names[idx] for idx in valid_indices[:valid_features]]
            
            plt.bar(range(valid_features), plot_importances,
                    color="b", yerr=plot_std, align="center")
            plt.xticks(range(valid_features), plot_names, rotation=90)
            plt.xlim([-1, valid_features])
            plt.tight_layout()
            
            # Save the plot
            output_dir = os.path.join(os.path.dirname(self.metrics_file))
            plt.savefig(os.path.join(output_dir, 'feature_importance_permutation.png'))
            plt.close()
            
            print(f"\nFeature importance plot saved to {os.path.join(output_dir, 'feature_importance_permutation.png')}")
    
    def print_classification_report(self):
        """
        Print classification report
        """
        if not hasattr(self, 'y_pred'):
            print("No predictions available. Train the classifier first.")
            return
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        auc_roc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        # Print classification report with additional metrics
        print("\nClassification Report:")
        print("-" * 60)
        print(classification_report(self.y_test, self.y_pred, target_names=['Noise', 'Whistle']))
        print("\nAdditional Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"AUC-ROC:   {auc_roc:.4f}")
        
        # Plot ROC curve with AUC score
        self.plot_roc_curve()
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations
        """
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.print_classification_report()
    
    def save_misclassified_to_csv(self, misclassified_indices=None):
        """Save basic information about misclassified examples to CSV
        
        Args:
            misclassified_indices: Optional list of indices to save. If None,
                                 finds misclassified examples from test set
        """
        if misclassified_indices is None:
            # Get predictions on test set if not already done
            if not hasattr(self, 'y_pred'):
                self.y_pred = self.clf.predict(self.X_test)
            misclassified_indices = np.where(self.y_pred != self.y_test)[0]
        
        # Ensure we're only working with actual misclassifications
        misclassified_indices = np.array([idx for idx in misclassified_indices 
                                        if idx < len(self.y_test) and self.y_pred[idx] != self.y_test[idx]])
        
        # Create basic data with essential information
        misclassified_data = pd.DataFrame({
            'file_index': self.index[misclassified_indices],
            'true_label': self.y_test[misclassified_indices],
            'predicted_label': self.y_pred[misclassified_indices],
            'prediction_probability': self.y_pred_proba[misclassified_indices]
        })
        
        # Save to CSV
        output_dir = os.path.join(os.path.dirname(self.metrics_file))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'misclassified_examples.csv')
        misclassified_data.to_csv(output_file, index=False)
        
        print(f"\nMisclassified examples saved to {output_file}")
        print(f"Total misclassified examples: {len(misclassified_indices)}")
        
        return misclassified_indices
    
    def analyze_misclassifications(self):
        """
        Analyze misclassified examples
        """
        if not hasattr(self, 'y_pred') or not hasattr(self, 'y_test'):
            print("No predictions available. Train the classifier first.")
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
        plt.axvline(x=0.5, color='r', linestyle='--', label='Decision threshold')
        plt.title('Prediction Probabilities for Misclassified Examples')
        plt.xlabel('Probability of Whistle')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(self.metrics_file))
        plt.savefig(os.path.join(output_dir, 'misclassified_probabilities.png'))
        plt.close()
        
        print(f"Misclassified probabilities plot saved to {os.path.join(output_dir, 'misclassified_probabilities.png')}")
        
        # Analyze feature values for misclassified examples
        misclassified_features = self.X_test[misclassified_indices]
        
        # Get feature importance
        if hasattr(self.clf, 'feature_importances_'):
            importances = self.clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Get top 5 important features
            top_features_idx = indices[:5]
            top_features_names = [self.feature_names[i] for i in top_features_idx]
            
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
            plt.savefig(os.path.join(output_dir, 'misclassified_feature_distributions.png'))
            plt.close()
            
            print(f"Misclassified feature distributions plot saved to {os.path.join(output_dir, 'misclassified_feature_distributions.png')}")
        
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
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_percent.png'))
        plt.close()
        
        print(f"Percentage confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix_percent.png')}")
        
        # Save misclassified examples to CSV
        self.save_misclassified_to_csv(misclassified_indices)
        
        # Return misclassified indices for further analysis
        return misclassified_indices, fp_indices, fn_indices
    
    def save_model(self, output_dir=None):
        """
        Save the trained model and optimal threshold
        
        Args:
            output_dir: Directory to save the model. If None, uses the metrics file directory.
        """
        if not hasattr(self, 'xgb_pipeline'):
            print("No trained model available. Train the classifier first.")
            return
            
        if output_dir is None:
            output_dir = os.path.dirname(self.metrics_file)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the XGBoost model
        model_path = os.path.join(output_dir, 'dolphin_classifier.json')
        self.xgb_pipeline.named_steps['classifier'].save_model(model_path)
        
        # Save the preprocessing components using joblib
        imputer_path = os.path.join(output_dir, 'imputer.joblib')
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(self.xgb_pipeline.named_steps['imputer'], imputer_path)
        joblib.dump(self.xgb_pipeline.named_steps['scaler'], scaler_path)
        
        # Convert NumPy types to native Python types
        def convert_to_python_type(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save the threshold and other metadata
        metadata = {
            'optimal_threshold': convert_to_python_type(
                self.optimal_threshold if hasattr(self, 'optimal_threshold') else 0.5
            ),
            'threshold_optimization': self.threshold_optimization,
            'feature_names': [convert_to_python_type(name) for name in self.feature_names],
            'skip_time_metrics': self.skip_time_metrics
        }
        
        metadata_path = os.path.join(output_dir, 'dolphin_classifier_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
            
        print(f"Model components saved to {output_dir}:")
        print(f"- XGBoost model: {model_path}")
        print(f"- Imputer: {imputer_path}")
        print(f"- Scaler: {scaler_path}")
        print(f"- Metadata: {metadata_path}")
        print(f"- Optimal threshold: {metadata['optimal_threshold']}")
        
    def load_model(self, model_dir=None):
        """
        Load a trained model and its metadata
        
        Args:
            model_dir: Directory containing the model files. If None, uses the metrics file directory.
        """
        if model_dir is None:
            model_dir = os.path.dirname(self.metrics_file)
            
        model_path = os.path.join(model_dir, 'dolphin_classifier.json')
        metadata_path = os.path.join(model_dir, 'dolphin_classifier_metadata.json')
        imputer_path = os.path.join(model_dir, 'imputer.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        required_files = {
            'Model': model_path,
            'Metadata': metadata_path,
            'Imputer': imputer_path,
            'Scaler': scaler_path
        }
        
        # Check if all required files exist
        missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
        if missing_files:
            print(f"Missing required files: {', '.join(missing_files)}")
            print("Train and save the model first.")
            return False
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)
            
        # Update instance attributes
        self.optimal_threshold = metadata.get('optimal_threshold', 0.5)  # Use get() with default
        self.threshold_optimization = metadata.get('threshold_optimization', False)
        self.feature_names = metadata.get('feature_names', [])
        self.skip_time_metrics = metadata.get('skip_time_metrics', True)
        
        # Load preprocessing components
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        
        # Create and load the XGBoost model
        xgb_clf = xgb.XGBClassifier()
        xgb_clf.load_model(model_path)
        
        # Create the pipeline with loaded components
        self.xgb_pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('classifier', xgb_clf)
        ])
        
        # Store classifier reference
        self.clf = xgb_clf
        
        print(f"Model and components loaded from {model_dir}")
        print(f"Using {'optimal' if self.threshold_optimization else 'default'} threshold: {self.optimal_threshold:.4f}")
        return True

    def verify_misclassified_examples(self):
        """
        Load and analyze the saved misclassified examples from CSV.
        Verifies that the saved examples are actually misclassified and counts them by type.
        """
        output_dir = os.path.dirname(self.metrics_file)
        csv_path = os.path.join(output_dir, 'misclassified_examples.csv')
        
        if not os.path.exists(csv_path):
            print(f"Error: Misclassified examples file not found at {csv_path}")
            return
            
        # Load the saved misclassified examples
        print(f"\nLoading misclassified examples from {csv_path}")
        misclassified_df = pd.read_csv(csv_path)
        
        # Count total examples
        total_examples = len(misclassified_df)
        
        # Count by type
        false_positives = np.sum((misclassified_df['predicted_label'] == 1) & 
                                (misclassified_df['true_label'] == 0))
        false_negatives = np.sum((misclassified_df['predicted_label'] == 0) & 
                                (misclassified_df['true_label'] == 1))
        
        # Verify all examples are actually misclassified
        actually_misclassified = np.sum(misclassified_df['predicted_label'] != misclassified_df['true_label'])
        
        print("\nMisclassified Examples Analysis:")
        print("-" * 40)
        print(f"Total examples in file: {total_examples}")
        print(f"Actually misclassified: {actually_misclassified}")
        print(f"False positives (noise classified as whistle): {false_positives}")
        print(f"False negatives (whistle classified as noise): {false_negatives}")
        
        # Verify consistency
        if actually_misclassified != total_examples:
            print("\n! Warning: Not all examples in the file are misclassified")
            print(f"  {total_examples - actually_misclassified} examples appear to be correctly classified")
        
        if actually_misclassified != (false_positives + false_negatives):
            print("\n! Warning: Inconsistency in misclassification counts")
            print(f"  Sum of FP and FN ({false_positives + false_negatives}) doesn't match total misclassified ({actually_misclassified})")
        
        # Analyze prediction probabilities
        print("\nPrediction Probability Analysis:")
        print("-" * 40)
        probs = misclassified_df['prediction_probability']
        print(f"Mean probability: {probs.mean():.4f}")
        print(f"Std deviation: {probs.std():.4f}")
        print(f"Min probability: {probs.min():.4f}")
        print(f"Max probability: {probs.max():.4f}")
        
        # Print file indices summary
        print("\nFile Indices Summary:")
        print("-" * 40)
        print(f"First few file indices: {', '.join(map(str, misclassified_df['file_index'].head()))}")
        print(f"Number of unique files: {misclassified_df['file_index'].nunique()}")
        
        return {
            'total': total_examples,
            'actually_misclassified': actually_misclassified,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'mean_probability': probs.mean(),
            'std_probability': probs.std()
        }

    def tune_parameters(self, method='random', n_iter=10, cv=5):
        """
        Tune XGBoost classifier parameters using grid search or randomized search
        
        Args:
            method: Method to use for parameter tuning ('grid' or 'random')
            n_iter: Number of iterations for randomized search
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and best score
        """
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Define parameter grid for XGBoost
        xgb_param_grid = config.XGB_PARAM_GRID
        
        # Create pipeline for XGBoost
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(random_state=self.random_state))
        ])
        
        # Tune XGBoost parameters
        print("\n=== Tuning XGBoost Parameters ===")
        
        # For RandomizedSearchCV, we can use a custom implementation with progress bar
        if method == 'random':
            print(f"Performing {n_iter} random search iterations with {cv}-fold cross-validation...")
            
            # Generate random parameter combinations
            import itertools
            from sklearn.model_selection import ParameterSampler
            
            # Sample random parameter combinations
            param_list = list(ParameterSampler(
                xgb_param_grid, n_iter=n_iter, random_state=self.random_state
            ))
            
            # Initialize variables to track best parameters
            best_score = 0
            best_params = None
            best_estimator = None
            
            # Perform cross-validation for each parameter combination with progress bar
            for params in tqdm(param_list, desc="XGB Parameter Tuning"):
                # Set parameters
                xgb_pipeline.set_params(**params)
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    xgb_pipeline, X_train, y_train, 
                    cv=cv, scoring='accuracy', n_jobs=-1
                )
                
                # Calculate mean score
                mean_score = np.mean(cv_scores)
                
                # Update best parameters if better score found
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            # Train best model
            xgb_pipeline.set_params(**best_params)
            xgb_pipeline.fit(X_train, y_train)
            best_estimator = xgb_pipeline
            
            # Create a mock object to mimic RandomizedSearchCV interface
            class MockSearchCV:
                def __init__(self, best_estimator, best_params, best_score):
                    self.best_estimator_ = best_estimator
                    self.best_params_ = best_params
                    self.best_score_ = best_score
                    self.cv_results_ = {'params': param_list, 'mean_test_score': []}
                    
                    # Calculate mean test scores for parameter importance
                    for params in param_list:
                        xgb_pipeline.set_params(**params)
                        cv_scores = cross_val_score(
                            xgb_pipeline, X_train, y_train, 
                            cv=2, scoring='accuracy', n_jobs=-1
                        )
                        self.cv_results_['mean_test_score'].append(np.mean(cv_scores))
            
            xgb_search = MockSearchCV(best_estimator, best_params, best_score)
        else:  # grid
            xgb_search = GridSearchCV(
                xgb_pipeline, xgb_param_grid, cv=cv, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            xgb_search.fit(X_train, y_train)
        
        print(f"Best XGBoost parameters: {xgb_search.best_params_}")
        print(f"Best XGBoost CV score: {xgb_search.best_score_:.4f}")
        
        # Evaluate on test set
        xgb_best = xgb_search.best_estimator_
        xgb_pred = xgb_best.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"XGBoost test accuracy: {xgb_accuracy:.4f}")
        
        # Store best model and parameters
        self.best_xgb_model = xgb_best
        self.best_xgb_params = xgb_search.best_params_
        
        # Store predictions for analysis
        self.y_pred_proba = xgb_best.predict_proba(X_test)[:, 1]
        self.y_pred = xgb_pred
        self.y_test = y_test
        self.X_test = X_test
        
        # For feature importance
        self.clf = xgb_best.named_steps['classifier']
        
        # Plot parameter importance
        self.plot_parameter_importance(xgb_search)
        
        return {
            'xgb_params': xgb_search.best_params_,
            'xgb_score': xgb_search.best_score_,
            'xgb_accuracy': xgb_accuracy
        }
    
    def plot_parameter_importance(self, xgb_search):
        """
        Plot parameter importance from grid search results
        
        Args:
            xgb_search: GridSearchCV or RandomizedSearchCV object for XGBoost
        """
        # Function to extract parameter importance from search results
        def get_param_importance(search_results):
            # Get all results
            cv_results = search_results.cv_results_
            
            # Get parameter names
            param_names = [param for param in cv_results['params'][0].keys()]
            
            # Calculate importance for each parameter
            param_importance = {}
            
            for param in param_names:
                # Get unique values for this parameter
                unique_values = set()
                for params in cv_results['params']:
                    unique_values.add(params[param])
                
                # Skip if only one value was tried
                if len(unique_values) <= 1:
                    continue
                
                # Calculate mean score for each value
                value_scores = {}
                for value in unique_values:
                    # Find all results with this parameter value
                    indices = [i for i, params in enumerate(cv_results['params']) 
                              if params[param] == value]
                    
                    # Calculate mean score
                    mean_score = np.mean([cv_results['mean_test_score'][i] for i in indices])
                    value_scores[value] = mean_score
                
                # Calculate importance as range of scores
                importance = max(value_scores.values()) - min(value_scores.values())
                param_importance[param] = importance
            
            return param_importance
        
        # Get parameter importance
        xgb_importance = get_param_importance(xgb_search)
        
        # Plot XGBoost parameter importance
        if xgb_importance:
            plt.figure(figsize=(12, 6))
            plt.title("XGBoost Parameter Importance")
            
            # Sort parameters by importance
            sorted_params = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)
            param_names = [p[0].replace('classifier__', '') for p in sorted_params]
            importances = [p[1] for p in sorted_params]
            
            plt.bar(range(len(importances)), importances, align='center')
            plt.xticks(range(len(importances)), param_names, rotation=45)
            plt.xlabel('Parameter')
            plt.ylabel('Importance (score range)')
            plt.tight_layout()
            
            # Save the plot
            output_dir = os.path.join(os.path.dirname(self.metrics_file))
            plt.savefig(os.path.join(output_dir, 'xgb_parameter_importance.png'))
            plt.close()
            
            print(f"XGBoost parameter importance plot saved to {os.path.join(output_dir, 'xgb_parameter_importance.png')}")

    def train_with_optimized_params(self, small_test=False):
        """
        Train the classifier with optimized parameters
        
        Args:
            small_test: If True, use only a small subset of data for quick testing
            
        Returns:
            Trained XGBoost accuracy
        """
        # Use the optimized parameters from the tuning results
        xgb_params = config.XGB_PARAMS.copy()
        if small_test:
            xgb_params['n_estimators'] = 20

        print(f"Training with optimized parameters")
        print(f"XGBoost parameters: {xgb_params}")
        
        X_train, X_test, y_train, y_test = self.prepare_data(small_test=small_test)
        
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        
        # Create pipeline
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', xgb_clf)
        ])
        
        # Train model
        print("Training XGBoost Classifier...")
        xgb_pipeline.fit(X_train, y_train)
        xgb_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_pipeline.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"XGBoost accuracy: {xgb_accuracy:.4f}")
        
        # Store predictions and model
        self.xgb_pipeline = xgb_pipeline
        self.y_pred_proba = xgb_pred_proba
        self.y_pred = xgb_pred
        self.y_test = y_test
        self.X_test = X_test
        
        # For feature importance
        self.clf = xgb_pipeline.named_steps['classifier']
        
        return xgb_accuracy


def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate dolphin whistle classifier')
    parser.add_argument('--test', action='store_true', help='Run in test mode with small dataset')
    parser.add_argument('--tune', action='store_true', help='Perform parameter tuning')
    parser.add_argument('--optimize-threshold', action='store_true', help='Use threshold optimization')
    parser.add_argument('--include-time-metrics', action='store_true', help='Include time metrics in feature set')
    args = parser.parse_args()
    
    # Path to the metrics file
    metrics_file = os.path.join(os.path.dirname(__file__), config.OUTPUT_DIR, 'dolphin_train_metrics.csv')
    
    # Load metadata to get original training parameters
    metadata_path = os.path.join(os.path.dirname(metrics_file), 'dolphin_classifier_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)
            threshold_optimization = metadata.get('threshold_optimization', False)
            skip_time_metrics = not any(name.startswith(('centTime_', 'bwTime_', 'skewTime_', 'tvTime_')) 
                                     for name in metadata.get('feature_names', []))
    else:
        print("Warning: No metadata file found. Using default parameters.")
        threshold_optimization = args.optimize_threshold
        skip_time_metrics = not args.include_time_metrics
    
    # Create classifier with parameters from metadata
    classifier = DolphinClassifier(
        metrics_file, 
        threshold_optimization=threshold_optimization,
        skip_time_metrics=skip_time_metrics
    )
    
    if args.optimize_threshold:
        print("\nUsing threshold optimization")
    else:
        print("\nUsing default threshold (0.5)")
    
    if args.tune:
        # Tune parameters and get best parameters
        print("\n=== Tuning XGBoost Parameters ===")
        tuning_results = classifier.tune_parameters(
            method='random',
            n_iter=20 if not args.test else 10,
            cv=config.CV_FOLDS if not args.test else config.SMALL_TEST_CV_FOLDS
        )
        
        # Print tuning results
        print("\n=== Tuning Results ===")
        print(f"Best parameters found:")
        for param, value in tuning_results['xgb_params'].items():
            print(f"  {param}: {value}")
        print(f"Best cross-validation score: {tuning_results['xgb_score']:.4f}")
        print(f"Test set accuracy: {tuning_results['xgb_accuracy']:.4f}")
        
        print("\n=== Using Best Model from Tuning ===")
        # Store the best model in xgb_pipeline for saving
        classifier.xgb_pipeline = classifier.best_xgb_model
    else:
        # Train with pre-optimized parameters
        print("\n=== Training with Default Optimized Parameters ===")
        classifier.train_with_optimized_params(small_test=args.test)
    
    if not args.test:
        # Perform cross-validation (skip in test mode)
        print("\n=== 10-Fold Cross-Validation with XGBoost ===")
        classifier.cross_validate(n_folds=10)
    else:
        # Use fewer folds for test mode
        print("\n=== 3-Fold Cross-Validation with XGBoost (Test Mode) ===")
        classifier.cross_validate(n_folds=3, small_test=True)
    
    # Generate visualizations and print classification report (only once)
    print("\n=== Generating Visualizations and Metrics ===")
    classifier.generate_all_visualizations()
    
    # Save the trained model
    print("\n=== Saving Trained Model ===")
    classifier.save_model()
    
    # Store original test data
    original_X_test = classifier.X_test
    original_y_test = classifier.y_test
    
    # Create a new classifier instance for loading with same parameters from metadata
    print("\n=== Loading and Testing Saved Model ===")
    loaded_classifier = DolphinClassifier(
        metrics_file,
        threshold_optimization=threshold_optimization,
        skip_time_metrics=skip_time_metrics  # Use same parameters as training
    )
    
    # Load the saved model
    if loaded_classifier.load_model():
        # Make predictions using loaded model
        predictions, pred_proba = loaded_classifier.predict_with_threshold(
            original_X_test,
            threshold=loaded_classifier.optimal_threshold
        )
        
        # Store predictions and test data for misclassification analysis
        loaded_classifier.y_pred = predictions
        loaded_classifier.y_pred_proba = pred_proba  # Store prediction probabilities
        loaded_classifier.y_test = original_y_test
        loaded_classifier.index = classifier.index  # Ensure we have the original file indices
        
        # Save basic information about misclassified examples
        print("\n=== Saving Misclassified Examples from Loaded Model ===")
        misclassified_indices = loaded_classifier.save_misclassified_to_csv()
        
        # Count actual misclassifications
        actual_misclassified = np.sum(predictions != original_y_test)
        print(f"Found {actual_misclassified} misclassified examples")
        
        # Verify saved misclassified examples
        print("\n=== Verifying Saved Misclassified Examples ===")
        loaded_classifier.verify_misclassified_examples()

if __name__ == "__main__":
    main()
