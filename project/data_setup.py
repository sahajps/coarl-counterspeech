import os
from huggingface_hub import login
from datasets import load_dataset

def main():
    # Prompt user to log in to their Hugging Face account using a token
    print("Please log in to your Hugging Face account.")
    token = input("Enter your Hugging Face access token: ")
    
    # Log in to Hugging Face
    try:
        login(token=token)
        print("Login successful!")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # Load the dataset from Hugging Face
    dataset_name = "Aswini123/IntentCONANv2"
    print(f"Downloading dataset: {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    # Create the data directory if it doesn't exist
    os.makedirs("project/data/", exist_ok=True)
    os.makedirs("project/log/", exist_ok=True)

    # Save the train, test, and validation splits as CSV files
    try:
        dataset['train'].to_csv('project/data/train.csv', index=False)
        dataset['test'].to_csv('project/data/test.csv', index=False)
        dataset['validation'].to_csv('project/data/validation.csv', index=False)
        print("Dataset splits saved successfully!")
    except Exception as e:
        print(f"Failed to save dataset splits: {e}")

if __name__ == "__main__":
    main()