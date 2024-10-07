from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from transformers import AdamW
import torch
import json

def train_model(data_file):
    # Load the dataset
    with open(data_file, 'r') as file:
        data = json.load(file)

    # Load the tokenizer and model for RAG
    tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-base')
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-base', use_dummy_dataset=True)
    model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-base')

    # Prepare the data for RAG
    prompts = [entry['prompt'] for entry in data]
    completions = [entry['completion'] for entry in data]

    # Tokenize inputs
    input_encodings = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    labels_encodings = tokenizer(completions, return_tensors='pt', padding=True, truncation=True)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)  # Define the optimizer

    # Training logic
    model.train()
    for epoch in range(3):  # Number of epochs
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(input_ids=input_encodings['input_ids'], labels=labels_encodings['input_ids'])
        loss = outputs.loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")  # Print loss for monitoring

if __name__ == "__main__":
    data_file = 'data/annotated_data.json'
    train_model(data_file)