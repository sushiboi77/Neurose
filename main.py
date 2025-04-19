from openai import OpenAI
import os
from dotenv import load_dotenv
from colorama import init, Fore, Style
import numpy as np
import json
import random

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

def load_used_hooks():
    """Load previously used hooks from file"""
    try:
        with open('used_hooks.json', 'r', encoding='utf-8') as f:
            return json.load(f)  # Returns list of hooks
    except FileNotFoundError:
        return []

def save_used_hooks(hook):
    """Save the new hook while maintaining last 10 hooks"""
    used_hooks = load_used_hooks()
    used_hooks.append(hook)
    
    # Keep only the last 10 hooks
    if len(used_hooks) > 10:
        used_hooks = used_hooks[-10:]
        
    with open('used_hooks.json', 'w', encoding='utf-8') as f:
        json.dump(used_hooks, f, ensure_ascii=False, indent=2)

def get_top_hooks(analysis_text, num_hooks=5):
    # Load previously used hooks
    used_hooks = set(load_used_hooks())  # Convert to set for faster lookup
    
    with open('hook_embeddings.json', 'r', encoding='utf-8') as f:
        hook_embeddings = json.load(f)
    
    # Filter out previously used hooks
    available_hooks = [hook_data['hook'] for hook_data in hook_embeddings 
                      if hook_data['hook'] not in used_hooks]
    
    # Randomly select hooks
    if len(available_hooks) <= num_hooks:
        return available_hooks  # Return all available hooks if fewer than num_hooks
    else:
        return random.sample(available_hooks, num_hooks)

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

def detect_used_hook(output_text, provided_hooks):
    """Detect which hook was used in the output by comparing embeddings"""
    # Extract the hook from the output (text between [Hook] and [Body])
    try:
        hook_text = output_text.split("[Hook]")[1].split("[Body]")[0].strip()
        print(f"\nExtracted Hook Text:\n{'-'*50}")
        print(Fore.CYAN + hook_text + Style.RESET_ALL)
        print('-'*50)
        
        hook_embedding = get_embedding(hook_text)
        
        # Compare with the provided hooks
        similarities = []
        print(f"\nHook Similarity Comparison:\n{'-'*50}")
        for hook in provided_hooks:
            hook_embedding_compare = get_embedding(hook)
            similarity = cosine_similarity(hook_embedding, hook_embedding_compare)
            similarities.append((hook, similarity))
            print(Fore.MAGENTA + f"Score: {similarity:.4f} - {hook}" + Style.RESET_ALL)
        
        # Return the hook with highest similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print('-'*50)
        return similarities[0][0]  # Return the most similar hook
    except Exception as e:
        print(f"Error detecting used hook: {e}")
        return None

def shorten_script(incomplete_script, analysis):
    """Use another agent to shorten the script if it's too long"""
    with open('system messages/shorterner.txt', 'r', encoding='utf-8') as file:
        shortener_system_message = file.read()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": shortener_system_message},
            {"role": "user", "content": f"Incomplete script:\n{incomplete_script}\n\nAnalysis:\n{analysis}"}
        ],
        temperature=0.1,
        max_tokens=112,
        stream=False
    )
    
    return response.choices[0].message.content

def generate_script():
    # Load both parts of the system message
    with open('system messages/content.txt', 'r', encoding='utf-8') as file:
        content_part1 = file.read()
    with open('system messages/content1.txt', 'r', encoding='utf-8') as file:
        content_part2 = file.read()
    
    # Get user inputs
    print("Enter your PineScript code (press Enter when done):")
    pinescript_code = input().strip()
    
    print("\nEnter a description of your indicator (press Enter when done):")
    description = input().strip()
    
    # Step 1: Get analysis from first agent
    combined_input = f"Code:\n{pinescript_code}\n\nDescription:\n{description}"
    analysis = analyze_code(combined_input)
    
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
    full_prompt = f"Analysis:\n{analysis}\n\nCode:\n{pinescript_code}\n\nDescription:\n{description}"
    print("\nFull Prompt to Second Agent:\n" + "-"*50)
    print(Fore.YELLOW + full_prompt + Style.RESET_ALL)
    print("-"*50)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=112,
            top_p=0.33,
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
       
        # Keep shortening until we get a proper ending
        max_attempts = 3
        attempt = 0
        current_script = full_response
        
        while not (current_script.strip().endswith('AlgoAlpha.') or 
                  current_script.strip().endswith('by AlgoAlpha')):
            attempt += 1
            if attempt > max_attempts:
                print("\nFailed to generate properly sized script after multiple attempts.")
                break
                
            print(f"\nScript exceeded length limit. Shortening (Attempt {attempt})...\n" + "-"*50)
            current_script = shorten_script(current_script, analysis)
            print(Fore.WHITE + current_script + Style.RESET_ALL)
            print("-"*50)

        full_response = current_script

        # Detect which hook was used and save it
        used_hook = detect_used_hook(full_response, top_hooks)
        if used_hook:
            print(f"\nDetected Used Hook:\n{'-'*50}")
            print(Fore.YELLOW + used_hook + Style.RESET_ALL)
            print('-'*50)
            save_used_hooks(used_hook)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_script()