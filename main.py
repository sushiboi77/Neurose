from openai import OpenAI
import os
from dotenv import load_dotenv
from colorama import init, Fore, Style
import numpy as np
import json

# Initialize colorama
init()

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    """Get embedding for a given text using OpenAI's embedding model"""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def get_top_hooks(analysis_text, num_hooks=5):
    analysis_embedding = get_embedding(analysis_text)
    
    with open('hook_embeddings.json', 'r', encoding='utf-8') as f:
        hook_embeddings = json.load(f)
    
    similarities = []
    for hook_data in hook_embeddings:
        similarity = cosine_similarity(analysis_embedding, hook_data['embedding'])
        similarities.append((hook_data['hook'], similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [hook for hook, _ in similarities[:num_hooks]]

def analyze_code(pinescript_code):
    with open('system messages/analysis.txt', 'r', encoding='utf-8') as file:
        analysis_system_message = file.read()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": analysis_system_message},
            {"role": "user", "content": pinescript_code}
        ],
        temperature=0.1,
        stream=False
    )
    
    analysis = response.choices[0].message.content
    print(f"\nFirst Agent Analysis:\n{'-'*50}")
    print(Fore.BLUE + analysis + Style.RESET_ALL)
    print('-'*50)
    return analysis

def generate_script():
    # Load both parts of the system message
    with open('system messages/content.txt', 'r', encoding='utf-8') as file:
        content_part1 = file.read()
    with open('system messages/content1.txt', 'r', encoding='utf-8') as file:
        content_part2 = file.read()
    
    # Get user input
    print("Enter your PineScript code (press Enter when done):")
    pinescript_code = input().strip()
    
    # Step 1: Get analysis from first agent
    analysis = analyze_code(pinescript_code)
    
    # Step 2: Get top 5 relevant hooks
    top_hooks = get_top_hooks(analysis)
    print(f"\nTop 5 Most Relevant Hooks:\n{'-'*50}")
    print(Fore.RED + "\n".join(f"{i}. {hook}" for i, hook in enumerate(top_hooks, 1)) + Style.RESET_ALL)
    print('-'*50)
    
    # Step 3: Combine system message parts with hooks
    system_message = content_part1 + "\n".join(f"{i}. {hook}" for i, hook in enumerate(top_hooks, 1)) + "\n\n" + content_part2
    
    print("\nFinal System Message:\n" + "-"*50)
    print(Fore.GREEN + system_message + Style.RESET_ALL)
    print("-"*50)
    
    # Print the full prompt that will be sent to the second agent
    full_prompt = f"Analysis:\n{analysis}\n\nCode:\n{pinescript_code}"
    print("\nFull Prompt to Second Agent:\n" + "-"*50)
    print(Fore.YELLOW + full_prompt + Style.RESET_ALL)
    print("-"*50)
    
    try:
        # Create the streaming response with the second agent
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7, # Increased temperature for more creative responses
            stream=True
        )

        print("\nGenerated Script:")
        print("-" * 50)
        
        full_response = ""
        buffer = ""

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(Fore.WHITE + content + Style.RESET_ALL, end="", flush=True)
                full_response += content
        
        print("\n" + "-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_script()
