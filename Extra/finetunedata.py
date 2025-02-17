import json

def create_training_data():
    # Get the system message that will be used for all entries
    print("\nEnter the system message that will be used for all entries:")
    system_message = input().strip()
    
    training_data = []
    
    while True:
        print("\nEnter user message (or 'end123' to finish):")
        user_message = input().strip()
        
        # Check if user wants to end the session
        if user_message.lower() == 'end123':
            break
            
        print("\nEnter assistant message:")
        assistant_message = input().strip()
        
        # Create the message format for this conversation
        conversation = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        
        training_data.append(conversation)
    
    # Write to JSONL file
    with open('training_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in training_data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("\nTraining data has been saved to 'training_data.jsonl'")
    print(f"Number of conversations generated: {len(training_data)}")

if __name__ == "__main__":
    print("Welcome to the Fine-tuning Data Generator!")
    print("This tool will help you create JSONL training data.")
    print("Type 'end123' when you want to finish entering conversations.\n")
    create_training_data()
