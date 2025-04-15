## Hakaton

### Installation

To install required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Generate Training Metrics

This step processes training audio files and generates metrics for model training:

```bash
python generate_training_metrics2.py --audio-dir ./audio_train --csv-path ./train.csv --template-audio-dir ./templates/audio --output-file ./output/train_metrics.csv
```

Parameters:

- `--audio-dir`: Directory containing training WAV files
- `--csv-path`: Path to CSV file with training labels
- `--template-audio-dir`: Directory containing template audio files
- `--output-file`: Where to save the generated metrics
- `--test`: (Optional) Run in test mode with reduced dataset
- `--test-size`: (Optional) Number of files to process in test mode (default: 5)
- `--num-processes`: (Optional) Number of processes to use (default: number of CPU cores)

**Note**: For initial testing, you can use the `--test` flag with a small dataset:

```bash
python generate_training_metrics2.py --test --test-size 5 --audio-dir ./audio_train --csv-path ./train.csv
```

#### 2. Generate Testing Metrics

This step processes test audio files and generates metrics for evaluation:

```bash
python generate_testing_metrics.py --template-audio-dir ./template_audio --template-file ./templates/template_definitions.csv --test-audio-dir ./audio_test --output-file ./output/test_metrics.csv
```

Parameters:

- `--test-audio-dir`: Directory containing WAV files for testing
- `--output-file`: Where to save the generated metrics

**Note**: Ensure your test audio directory contains the WAV files

#### 3. Evaluate Algorithm

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

**Note**: The `--truth-file` parameter is optional

### Full Training Instructions

1. Generate training metrics:

```bash
python generate_training_metrics2.py --audio-dir ./hakaton/audio_train --csv-path ./hakaton/train.csv --output-file dolphin_train_metrics.csv --template-audio-dir ./template_audio
```

2. Train classifier:

```bash
python generate_training_metrics2.py --audio-dir ./hakaton/audio_train --csv-path ./hakaton/train.csv --output-file dolphin_train_metrics.csv --template-audio-dir ./template_audio
```

### Using Data Splitter

If your data is organized in folders as:

```
hakaton/
    whistles/
    noise/
```

The data splitter will:

1. Split your data into train/test sets (80/20 by default)
2. Create the following structure:
   ```
   hakaton/
       audio_train/    # Training files
       audio_test/     # Testing files
       train.csv       # Training labels
       test.csv        # Testing labels
   ```

To run the splitter:

```bash
python datasetsplitter.py
```

After splitting, proceed with the training instructions above.

Note: The splitter automatically:

- Creates necessary directories
- Copies files to train/test folders
- Generates CSV files with labels
- Applies preprocessing to audio files (bandpass filter 5-15kHz)
