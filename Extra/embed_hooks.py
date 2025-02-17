from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import numpy as np

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embedding(text):
    """Get embedding for a given text using OpenAI's embedding model"""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def load_and_embed_hooks():
    # Read hooks from file
    with open('Extra/ideal_hooks.txt', 'r', encoding='utf-8') as file:
        # Split by newlines and filter out empty lines
        hooks = [hook.strip() for hook in file.read().split('\n\n') if hook.strip()]
    
    # Get embeddings for each hook
    hook_embeddings = []
    for hook in hooks:
        embedding = get_embedding(hook)
        hook_embeddings.append({
            "hook": hook,
            "embedding": embedding
        })
    
    # Save embeddings to a JSON file
    with open('hook_embeddings.json', 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = [{
            "hook": item["hook"],
            "embedding": list(map(float, item["embedding"]))
        } for item in hook_embeddings]
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(hooks)} hooks and saved embeddings to hook_embeddings.json")
    return hook_embeddings

if __name__ == "__main__":
    hook_embeddings = load_and_embed_hooks() 