#!/usr/bin/env python3
"""
Script to calculate the average of Test Approx and Train Time values 
from training log files in the DETAILED RESULTS section.
"""

import re
import sys
import argparse


def extract_values_from_log(log_file_path):
    """Extract Test Approx and Train Time values from the log file."""
    test_approx_values = []
    train_time_values = []
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the DETAILED RESULTS section
    detailed_results_match = re.search(r'DETAILED RESULTS.*?\n(.*?)(?=\n\n|\n=|$)', content, re.DOTALL)
    if not detailed_results_match:
        print("DETAILED RESULTS section not found in the log file.")
        return [], []
    
    detailed_section = detailed_results_match.group(1)
    
    # Extract data lines (skip header and separator lines)
    lines = detailed_section.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('-') or 'Epoch' in line or 'Train Loss' in line:
            continue
        
        # Parse the data line
        # Expected format: Epoch  Train Loss   Edge Error   Train Time   Val Approx   Test Approx
        parts = line.split()
        if len(parts) >= 6:
            try:
                # Train Time is the 4th column (index 3)
                train_time = float(parts[3])
                train_time_values.append(train_time)
                
                # Test Approx is the last column, remove % if present
                test_approx_str = parts[-1].replace('%', '')
                test_approx = float(test_approx_str)
                test_approx_values.append(test_approx)
                
            except (ValueError, IndexError):
                continue
    
    return test_approx_values, train_time_values


def calculate_averages(log_file_path):
    """Calculate and display averages for Test Approx and Train Time."""
    test_approx_values, train_time_values = extract_values_from_log(log_file_path)
    
    if not test_approx_values and not train_time_values:
        print("No valid data found in the log file.")
        return
    
    print(f"Log file: {log_file_path}")
    print("=" * 50)
    
    if test_approx_values:
        avg_test_approx = sum(test_approx_values) / len(test_approx_values)
        print(f"Test Approx values found: {len(test_approx_values)}")
        print(f"Average Test Approx: {avg_test_approx:.2f}%")
    else:
        print("No Test Approx values found.")
    
    if train_time_values:
        avg_train_time = sum(train_time_values) / len(train_time_values)
        print(f"Train Time values found: {len(train_time_values)}")
        print(f"Average Train Time: {avg_train_time:.2f} seconds")
    else:
        print("No Train Time values found.")


def main():
    parser = argparse.ArgumentParser(description='Calculate averages of Test Approx and Train Time from training logs')
    parser.add_argument('log_file', help='Path to the training log file')
    
    args = parser.parse_args()
    
    try:
        calculate_averages(args.log_file)
    except FileNotFoundError:
        print(f"Error: File '{args.log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()