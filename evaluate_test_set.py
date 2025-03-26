import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from dolphin_detector import config
except ImportError:
    import config

class DolphinEvaluator:
    def __init__(self, metrics_file, mapping_file, model_dir=None, output_file=None, truth_file=None):
        """
        Initialize the evaluator
        
        Args:
            metrics_file: Path to the test metrics CSV file
            mapping_file: Path to the file mappings CSV (index to filename)
            model_dir: Directory containing the saved model (default: metrics file directory)
            output_file: Path to save the results CSV file (default: results.csv in model dir)
            truth_file: Path to a file containing truth labels for comparison (optional)
        """
        self.metrics_file = metrics_file
        self.mapping_file = mapping_file
        self.truth_file = truth_file
        
        # Use metrics file directory if no model directory specified
        if model_dir is None:
            self.model_dir = os.path.dirname(metrics_file)
        else:
            self.model_dir = model_dir
            
        # Set default output file if not specified
        if output_file is None:
            self.output_file = os.path.join(self.model_dir, 'results.csv')
        else:
            self.output_file = output_file
        
        # Load file mappings first
        self.load_file_mappings()
        
        # Load model and test data
        self.load_model()
        self.load_test_metrics()
        
        # Load truth data if provided
        if self.truth_file:
            self.load_truth_data()
        
    def load_file_mappings(self):
        """
        Load the file mappings from index to filename
        """
        print(f"Loading file mappings from {self.mapping_file}")
        mapping_df = pd.read_csv(self.mapping_file)
        
        # Check mapping file structure
        if len(mapping_df.columns) < 2:
            raise ValueError(f"Mapping file should have at least 2 columns: {self.mapping_file}")
        
        # Create a dictionary mapping from index to filename
        self.file_map = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))
        
        print(f"Loaded {len(self.file_map)} file mappings")
        
    def load_model(self):
        """
        Load the saved classifier model and metadata
        """
        model_path = os.path.join(self.model_dir, 'dolphin_classifier.json')
        metadata_path = os.path.join(self.model_dir, 'dolphin_classifier_metadata.json')
        imputer_path = os.path.join(self.model_dir, 'imputer.joblib')
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        
        required_files = {
            'Model': model_path,
            'Metadata': metadata_path,
            'Imputer': imputer_path,
            'Scaler': scaler_path
        }
        
        # Check if all required files exist
        missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            import json
            self.metadata = json.load(f)
            
        # Get parameters from metadata
        self.threshold = self.metadata.get('optimal_threshold', 0.5)
        self.skip_time_metrics = self.metadata.get('skip_time_metrics', True)
        self.feature_names = self.metadata.get('feature_names', [])
        
        # Load preprocessing components
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        
        # Create and load the XGBoost model
        xgb_clf = xgb.XGBClassifier()
        xgb_clf.load_model(model_path)
        
        # Create the pipeline with loaded components
        self.pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('classifier', xgb_clf)
        ])
        
        print(f"Loaded model from {self.model_dir}")
        print(f"Using threshold: {self.threshold:.4f}")
        print(f"Time metrics: {'excluded' if self.skip_time_metrics else 'included'}")
    
    def load_test_metrics(self):
        """
        Load test metrics from CSV file
        """
        print(f"Loading test metrics from {self.metrics_file}")
        data = pd.read_csv(self.metrics_file)
        
        # Extract indices (assuming they are in the second column)
        self.indices = np.array(data.iloc[:, 1])
        
        # Map indices to filenames using the file mapping
        self.filenames = []
        for idx in self.indices:
            if idx in self.file_map:
                self.filenames.append(self.file_map[idx])
            else:
                print(f"Warning: Index {idx} not found in file mapping. Using index as filename.")
                self.filenames.append(str(idx))
        
        self.filenames = np.array(self.filenames)
        
        # MODIFIED: Only check for truth in metrics file if truth_file was not provided
        # This ensures we only use explicitly provided truth data
        if self.truth_file is None and data.columns[0].lower() in ['truth', 'label', 'class']:
            print("Note: First column in metrics file appears to contain labels, but will be ignored")
            print("      Use --truth-file parameter if you want to evaluate against truth labels")
            
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
            time_metrics_end += 4 * config.MAX_TIME
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
        
        # Print dataset statistics
        self.n_samples = len(self.features)
        print(f"Loaded {self.n_samples} test samples")
    
    def load_truth_data(self):
        """
        Load truth data from a separate file for comparison with predictions
        """
        print(f"Loading truth data from {self.truth_file}")
        
        try:
            truth_df = pd.read_csv(self.truth_file)
            
            if 'fname' not in truth_df.columns and 'filename' not in truth_df.columns:
                # If no filename column, check if we have 'index' column that matches our indices
                if 'index' in truth_df.columns:
                    # Map by index
                    truth_dict = dict(zip(truth_df['index'], truth_df['label']))
                    self.truth_labels = []
                    for idx in self.indices:
                        if idx in truth_dict:
                            label_str = truth_dict[idx].lower()
                            label = 1 if label_str == 'whistles' or label_str == 'whistle' else 0
                            self.truth_labels.append(label)
                        else:
                            print(f"Warning: No truth label for index {idx}")
                            self.truth_labels.append(None)
                else:
                    raise ValueError("Truth file must have 'fname'/'filename' or 'index' column")
            else:
                # Map by filename
                filename_col = 'fname' if 'fname' in truth_df.columns else 'filename'
                truth_dict = dict(zip(truth_df[filename_col], truth_df['label']))
                self.truth_labels = []
                for fname in self.filenames:
                    base_fname = os.path.basename(fname)
                    if base_fname in truth_dict:
                        label_str = truth_dict[base_fname].lower()
                        label = 1 if label_str == 'whistles' or label_str == 'whistle' else 0
                        self.truth_labels.append(label)
                    elif fname in truth_dict:
                        label_str = truth_dict[fname].lower()
                        label = 1 if label_str == 'whistles' or label_str == 'whistle' else 0
                        self.truth_labels.append(label)
                    else:
                        print(f"Warning: No truth label for file {fname}")
                        self.truth_labels.append(None)
            
            # Convert to NumPy array
            self.truth = np.array(self.truth_labels)
            
            # Count valid labels
            valid_labels = [label for label in self.truth_labels if label is not None]
            n_positive = sum(1 for label in valid_labels if label == 1)
            n_negative = sum(1 for label in valid_labels if label == 0)
            
            print(f"Loaded {len(valid_labels)} valid truth labels from truth file")
            print(f"Truth distribution: {n_positive} whistles, {n_negative} noise samples")
            
            self.truth_source = "truth_file"
            
        except Exception as e:
            print(f"Error loading truth data: {e}")
            self.truth = None
    
    def evaluate(self):
        """
        Evaluate the model on test data
        """
        print(f"Evaluating model on {self.n_samples} test samples...")
        
        # Get prediction probabilities
        self.pred_proba = self.pipeline.predict_proba(self.features)[:, 1]
        
        # Apply threshold
        self.predictions = (self.pred_proba >= self.threshold).astype(int)
        
        # Map predictions to labels (0 -> "noise", 1 -> "whistles")
        self.labels = ["noise" if pred == 0 else "whistles" for pred in self.predictions]
        
        # Create results dataframe
        self.results = pd.DataFrame({
            'fname': self.filenames,
            'label': self.labels,
            'probability': self.pred_proba,
            'index': self.indices
        })
        
        # Print prediction statistics
        n_whistles = sum(self.predictions == 1)
        n_noise = sum(self.predictions == 0)
        print(f"Predictions: {n_whistles} whistles ({n_whistles/self.n_samples:.1%}), "
              f"{n_noise} noise ({n_noise/self.n_samples:.1%})")
        
        # MODIFIED: Only calculate metrics if truth file was explicitly provided
        if self.truth_file is not None and hasattr(self, 'truth') and self.truth is not None:
            # Filter out None values in truth data if any
            valid_indices = [i for i, t in enumerate(self.truth) if t is not None]
            if len(valid_indices) < len(self.truth):
                print(f"Note: {len(self.truth) - len(valid_indices)} samples have no truth label and will be excluded from evaluation")
            
            truth_array = np.array([self.truth[i] for i in valid_indices])
            pred_array = np.array([self.predictions[i] for i in valid_indices])
            
            # Calculate metrics
            accuracy = accuracy_score(truth_array, pred_array)
            precision = precision_score(truth_array, pred_array, zero_division=0)
            recall = recall_score(truth_array, pred_array, zero_division=0)
            f1 = f1_score(truth_array, pred_array, zero_division=0)
            
            # Print evaluation metrics
            print(f"\nEvaluation Metrics (on {len(valid_indices)} samples with truth labels):")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            
            # Print confusion matrix
            cm = confusion_matrix(truth_array, pred_array)
            print("\nConfusion Matrix:")
            print("                Predicted")
            print("                Noise  Whistles")
            print(f"Actual Noise    {cm[0,0]:5d}  {cm[0,1]:8d}")
            print(f"       Whistles {cm[1,0]:5d}  {cm[1,1]:8d}")
            
            print("\nClassification Report:")
            print(classification_report(truth_array, pred_array, 
                                       target_names=['Noise', 'Whistles']))
            
            # Add truth labels to results
            self.results['truth_label'] = 'unknown'
            
            # Update truth labels for valid indices
            for i, idx in enumerate(valid_indices):
                truth_val = truth_array[i]
                truth_label = "noise" if truth_val == 0 else "whistles"
                self.results.loc[idx, 'truth_label'] = truth_label
        
        return self.results
    
    def save_results(self, include_probability=False, include_index=False, include_truth=True):
        """
        Save the results to a CSV file
        
        Args:
            include_probability: Whether to include the prediction probability in the output
            include_index: Whether to include the original index in the output
            include_truth: Whether to include truth labels in the output (if available)
        """
        if not hasattr(self, 'results'):
            raise ValueError("No results available. Run evaluate() first.")
        
        # Select columns to save
        columns = ['fname', 'label']
        if include_probability:
            columns.append('probability')
        if include_index:
            columns.append('index')
        if include_truth and self.truth_file is not None and 'truth_label' in self.results.columns:
            columns.append('truth_label')
            
        output_df = self.results[columns]
        
        # Save results to CSV
        output_df.to_csv(self.output_file, index=False)
        
        print(f"Results saved to {self.output_file}")
        print(f"Output includes columns: {', '.join(columns)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate dolphin whistle classifier on test set')
    parser.add_argument('--metrics-file', type=str, required=True,
                        help='Path to the test metrics CSV file')
    parser.add_argument('--mapping-file', type=str, required=True,
                        help='Path to the file mappings CSV (index to filename)')
    parser.add_argument('--model-dir', type=str, default=None, 
                        help='Directory containing the saved model (default: metrics file directory)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save the results CSV file (default: results.csv in model directory)')
    parser.add_argument('--truth-file', type=str, default=None,
                        help='Path to a file containing truth labels for comparison')
    parser.add_argument('--include-probability', action='store_true',
                        help='Include prediction probability in the output CSV')
    parser.add_argument('--include-index', action='store_true',
                        help='Include original index in the output CSV')
    parser.add_argument('--include-truth', action='store_true',
                        help='Include truth labels in the output CSV (if available)')
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = DolphinEvaluator(
            metrics_file=args.metrics_file,
            mapping_file=args.mapping_file,
            model_dir=args.model_dir,
            output_file=args.output_file,
            truth_file=args.truth_file
        )
        
        # Evaluate model
        evaluator.evaluate()
        
        # Save results
        evaluator.save_results(
            include_probability=args.include_probability,
            include_index=args.include_index,
            include_truth=args.include_truth
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
