import os
import warnings
from transformers import pipeline

# Suppress standard Hugging Face warnings for a cleaner terminal output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

print("🧠 Loading AI Models into memory... (This will take a few seconds)")
# Load the 28-emotion RoBERTa model
roberta_model = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')
# Load the 6-emotion BERT model
bert_model = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion')
print("✅ Models loaded successfully!\n")

def analyze_emotion():
    print("=" * 60)
    print(" 🎙️  Business Call Emotion Analyzer CLI")
    print("=" * 60)
    
    while True:
        # Step 1: Ask for user input
        print("\n" + "-" * 60)
        user_text = input("📝 Enter the text to analyze (or type 'exit' to quit):\n> ")
        
        # Exit condition
        if user_text.strip().lower() == 'exit':
            print("\nShutting down analyzer. Goodbye! 👋")
            break
            
        # Handle empty inputs
        if not user_text.strip():
            print("⚠️ Please enter some valid text before proceeding.")
            continue

        # Step 2: Ask for model choice
        print("\n🤖 Which model should process this text?")
        print("  [1] RoBERTa (28 fine-grained business emotions)")
        print("  [2] BERT (6 standard emotions)")
        
        model_choice = input("> Enter 1 or 2: ").strip()
        
        # Step 3: Pass to the selected model and output the result
        if model_choice == '1':
            print("🏃‍♂️ Running RoBERTa inference...")
            result = roberta_model(user_text)[0]
            print("\n" + "*" * 40)
            print(f"🎯 PREDICTED EMOTION : {result['label'].upper()}")
            print(f"📊 CONFIDENCE SCORE  : {result['score']:.4f}")
            print("*" * 40)
            
        elif model_choice == '2':
            print("🏃‍♂️ Running BERT inference...")
            result = bert_model(user_text)[0]
            print("\n" + "*" * 40)
            print(f"🎯 PREDICTED EMOTION : {result['label'].upper()}")
            print(f"📊 CONFIDENCE SCORE  : {result['score']:.4f}")
            print("*" * 40)
            
        else:
            print("⚠️ Invalid choice. Please type exactly '1' or '2'.")

if __name__ == "__main__":
    analyze_emotion()