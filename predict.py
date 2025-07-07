#!/usr/bin/env python3
"""
Simple script to predict resume categories.
Usage: python predict.py "resume text here"
"""

import sys
import joblib
import pickle
import pandas as pd
from data_cleaning.cleaner import clean_text

def load_model():
    """Load the trained model and vectorizer."""
    try:
        model = joblib.load('saved_models/resume_classifier.pkl')
        with open('saved_models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        print("❌ Model files not found! Please train the model first.")
        print("Run: python train_model.py")
        return None, None

def predict_category(text, model, vectorizer):
    """Predict category for given text."""
    # Clean text
    cleaned_text = clean_text(text)
    
    # Vectorize
    features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return prediction, probabilities

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"resume text here\"")
        print("Example: python predict.py \"Experienced software developer with Python skills\"")
        return
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    # Get input text
    resume_text = sys.argv[1]
    
    # Make prediction
    prediction, probabilities = predict_category(resume_text, model, vectorizer)
    
    # Display results
    print(f"\n📄 Resume Text: {resume_text}")
    print(f"🎯 Predicted Category: {prediction}")
    print(f"📊 Confidence: {max(probabilities):.2%}")
    
    print("\n📈 Category Probabilities:")
    for i, prob in enumerate(probabilities):
        if prob > 0.01:  # Show only significant probabilities
            print(f"  - {model.classes_[i]}: {prob:.2%}")

if __name__ == "__main__":
    main() 