## Hakaton

### Installation

To install required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Generate Testing Metrics

This step processes test audio files and generates metrics for evaluation:

```bash
python generate_testing_metrics.py --template-audio-dir ./template_audio --template-file ./templates/template_definitions.csv --test-audio-dir ./audio_test --output-file ./output/test_metrics.csv
```

Parameters:

- `--test-audio-dir`: Directory containing WAV files for testing
- `--output-file`: Where to save the generated metrics

**Note**: Ensure your test audio directory contains the WAV files

#### 2. Evaluate Algorithm

After generating metrics, evaluate the algorithm's performance:

```bash
python evaluate_test_set.py --metrics-file output/test_metrics.csv --mapping-file output/test_file_mapping.csv --model-dir ./model --output-file output/result.csv --include-probability --include-index --truth-file=output/test1.csv --include-truth
```

Parameters:

- `--metrics-file`: CSV file with test metrics (output from previous step)
- `--mapping-file`: File mapping test files to their identifiers (also output of previous step)
- `--model-dir`: Directory containing the model
- `--output-file`: Where to save evaluation results
- `--include-probability`: Include probability scores in results
- `--include-index`: Include index values in results
- `--truth-file`: Optional file containing ground truth values
- `--include-truth`: Include truth values in output (when truth file is provided)

**Note**: The `--truth-file` parameter is optional but recommended for comprehensive evaluation.
