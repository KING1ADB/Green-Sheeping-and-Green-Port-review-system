import json
import os

def preprocess_data(input_file, output_file):
    # Load the data with explicit encoding
    with open(input_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    # Process and save the data
    processed_data = []
    for entry in data:
        processed_data.append({
            'prompt': entry['prompt'],
            'completion': entry['completion']
        })

    # Save to a new file with explicit encoding
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = os.path.join('data', 'training_data.jsonl')
    output_file = os.path.join('data', 'processed_data.json')
    preprocess_data(input_file, output_file)