from predict.py import TopicPredictor

def test_inference():
    print("Iniciando prueba de inferencia...")
    try:
        predictor = TopicPredictor()
        
        # Frases de prueba para categorías típicas de HR
        test_phrases = [
            "How can I change my bank account for the next payment?", # Payroll
            "What is the company policy for maternity leave?",       # Benefits
            "I want to report an issue with my direct supervisor",     # Performance / Employee Relations
            "Can I get a copy of my last W2 form?"                    # Tax Services
        ]
        
        print(f"{'Frase':<60} | {'Predicción':<20} | {'Confianza':<10}")
        print("-" * 95)
        
        for phrase in test_phrases:
            result = predictor.predict(phrase)
            topic = result.get('topic', 'N/A')
            conf = result.get('confidence', 0.0)
            print(f"{phrase:<60} | {topic:<20} | {conf:.2f}")
            
    except Exception as e:
        print(f"Error durante la prueba: {e}")

if __name__ == "__main__":
    test_inference()
