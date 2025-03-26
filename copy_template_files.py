"""
copy_template_files.py

This script copies template audio files from source directory to a template directory
based on the template_definitions.csv file.
"""

import os
import shutil
import pandas as pd
import argparse

def copy_template_files(template_file, source_dir, dest_dir):
    """
    Copy template audio files from source to destination directory.
    
    Args:
        template_file: Path to template_definitions.csv
        source_dir: Directory containing source audio files
        dest_dir: Directory to copy template files to
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Read template definitions
    print(f"Reading template definitions from {template_file}")
    template_df = pd.read_csv(template_file)
    
    # Get unique audio files needed for templates
    template_files = set(template_df['fname'].unique())
    print(f"Found {len(template_files)} unique template files to copy")
    
    # Copy each file
    copied = 0
    failed = 0
    for fname in template_files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {fname}")
            copied += 1
        else:
            print(f"ERROR: Source file not found: {fname}")
            failed += 1
    
    print(f"\nSummary:")
    print(f"- Total files to copy: {len(template_files)}")
    print(f"- Successfully copied: {copied}")
    print(f"- Failed to copy: {failed}")
    print(f"\nTemplate files copied to: {dest_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Copy template audio files to a dedicated directory')
    parser.add_argument('--template-file', type=str, required=True,
                       help='Path to template_definitions.csv')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Directory containing source audio files')
    parser.add_argument('--dest-dir', type=str, required=True,
                       help='Directory to copy template files to')
    return parser.parse_args()

def main():
    args = parse_args()
    copy_template_files(args.template_file, args.source_dir, args.dest_dir)

if __name__ == "__main__":
    main() 
