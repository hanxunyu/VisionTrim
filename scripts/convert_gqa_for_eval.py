import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

all_answers = []
with open(args.src, 'r') as f:
    # First read the entire file content
    content = f.read().strip()
    
    try:
        # Try to parse as a single JSON object
        data = json.loads(content)
        if isinstance(data, dict):
            question_id = data['question_id']
            text = data['text'].rstrip('.').lower()
            all_answers.append({"questionId": question_id, "prediction": text})
    except json.JSONDecodeError:
        # If failed, try parsing line by line
        for line_idx, line in enumerate(content.split('\n')):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                res = json.loads(line)
                question_id = res['question_id']
                text = res['text'].rstrip('.').lower()
                all_answers.append({"questionId": question_id, "prediction": text})
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_idx + 1}: {e}")
                print(f"Problematic line content: {line[:100]}...")  # Only print first 100 characters

# Check if any data was successfully parsed
if not all_answers:
    print("Warning: No data was successfully parsed")
else:
    # Write results
    with open(args.dst, 'w') as f:
        json.dump(all_answers, f)
