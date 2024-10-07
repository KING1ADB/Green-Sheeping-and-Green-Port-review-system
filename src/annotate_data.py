import openai
import json
import os
import time

# Set your OpenAI API key here
openai.api_key = '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'

def annotate_data(data_file, output_file):
    # Load the data with explicit encoding
    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    annotated_data = []

    for entry in data:
        prompt = entry['prompt']

        max_retries = 5  # Maximum number of retries for rate limit
        retries = 0  # Current retry count

        while retries < max_retries:
            try:
                message = {"role": "user", "content": f"Identify named entities in the following text: {prompt}"}

                # Call OpenAI's chat completion API
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[message]
                )

                # Access the content of the response
                entities = response.choices[0].message.content
                annotated_data.append({
                    'prompt': prompt,
                    'completion': entry['completion'],
                    'entities': entities
                })

                # Sleep for a second after each successful request
                time.sleep(2)  # Throttle requests
                break  # Exit the loop if the request was successful

            except openai.RateLimitError:
                wait_time = 2 ** retries + 5  # Increase wait time with a base of 5 seconds
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1  # Increment retry count

            except Exception as e:
                print(f"An error occurred: {e}")
                break  # Exit the loop on other errors

        if retries == max_retries:
            print(f"Failed to process entry after {max_retries} retries: {prompt}")

    # Save annotated data with explicit encoding
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(annotated_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    data_file = os.path.join('data', 'processed_data.json')
    output_file = os.path.join('data', 'annotated_data.json')
    annotate_data(data_file, output_file)