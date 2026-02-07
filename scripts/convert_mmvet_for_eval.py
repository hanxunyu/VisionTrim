import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

cur_result = {}

# Add error handling and file format checking
with open(args.src, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        try:
            data = json.loads(line)
            qid = data['question_id']
            cur_result[f'v1_{qid}'] = data['text']
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")
            print(f"Problematic line content: {line[:100]}...")  # Print first 100 characters of the problematic line
            continue

# Check if any data was successfully parsed
if not cur_result:
    print("Warning: No data was successfully parsed")
else:
    # Write results
    with open(args.dst, 'w') as f:
        json.dump(cur_result, f, indent=2)
    print(f"Successfully processed {len(cur_result)} entries")
