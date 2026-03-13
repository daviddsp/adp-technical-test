from predict import TopicPredictor
import sys

def main():
    print("--- ADP HR Topic Classifier (CLI Mode) ---")
    try:
        predictor = TopicPredictor(model_dir="./saved_model", topics_path="data/available_topics.csv")
    except FileNotFoundError:
        print("Error: Model artifacts not found. Please run 'uv run train.py' first.")
        return

    print("Type your HR query below (or type 'exit' to quit):")
    while True:
        try:
            query = input("\n>> ").strip()
            if not query or query.lower() == 'exit':
                break
            
            result = predictor.predict(query)
            
            if result['status'] == 'success':
                print(f"Topic: {result['topic']} (Confidence: {result['confidence']:.2f})")
            else:
                print(f"Status: {result['status']} (The operation is not supported - Confidence: {result['confidence']:.2f})")
                
        except KeyboardInterrupt:
            break
    print("\nExiting. Good luck with the assignment!")

if __name__ == "__main__":
    main()
