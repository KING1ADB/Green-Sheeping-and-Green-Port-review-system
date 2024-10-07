from transformers import RagTokenizer, RagSequenceForGeneration
import json

def evaluate_model(prompt):
    tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-base')
    model = RagSequenceForGeneration.from_pretrained('./results')

    # Encode the prompt for RAG
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate a response
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    prompt = "What is the article titled 'Evolution of green shipping research: themes and methods' about?"
    print(evaluate_model(prompt))