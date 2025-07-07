#!/usr/bin/env python3
"""
Batch prediction script for multiple resume files.
Usage: python batch_predict.py data/raw_resumes/
"""

import os
import sys
import joblib
import pickle
import pandas as pd
from data_cleaning.cleaner import clean_text
from resume_parser.parser import parse_resume_file

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
    cleaned_text = clean_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities

def process_directory(directory_path, model, vectorizer):
    """Process all resume files in a directory."""
    results = []
    supported_extensions = ['.pdf', '.docx', '.txt']
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return results
    
    files = [f for f in os.listdir(directory_path) 
             if any(f.lower().endswith(ext) for ext in supported_extensions)]
    
    if not files:
        print(f"❌ No supported files found in {directory_path}")
        return results
    
    print(f"📁 Processing {len(files)} files in {directory_path}...")
    
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        print(f"  Processing: {filename}")
        
        try:
            # Parse file
            text = parse_resume_file(file_path)
            
            if text:
                # Make prediction
                prediction, probabilities = predict_category(text, model, vectorizer)
                confidence = max(probabilities)
                
                results.append({
                    'filename': filename,
                    'predicted_category': prediction,
                    'confidence': confidence,
                    'text_length': len(text)
                })
                
                print(f"    → {prediction} (confidence: {confidence:.2%})")
            else:
                print(f"    → Failed to extract text")
                results.append({
                    'filename': filename,
                    'predicted_category': 'ERROR',
                    'confidence': 0.0,
                    'text_length': 0
                })
                
        except Exception as e:
            print(f"    → Error: {str(e)}")
            results.append({
                'filename': filename,
                'predicted_category': 'ERROR',
                'confidence': 0.0,
                'text_length': 0
            })
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_predict.py <directory_path>")
        print("Example: python batch_predict.py data/raw_resumes/")
        return
    
    directory_path = sys.argv[1]
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    # Process files
    results = process_directory(directory_path, model, vectorizer)
    
    if results:
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        
        # Save results
        output_file = 'batch_predictions.csv'
        df_results.to_csv(output_file, index=False)
        
        # Display summary
        print(f"\n📊 Summary:")
        print(f"Total files processed: {len(results)}")
        print(f"Successful predictions: {len(df_results[df_results['predicted_category'] != 'ERROR'])}")
        print(f"Errors: {len(df_results[df_results['predicted_category'] == 'ERROR'])}")
        
        if len(df_results[df_results['predicted_category'] != 'ERROR']) > 0:
            print(f"\n🎯 Category Distribution:")
            category_counts = df_results[df_results['predicted_category'] != 'ERROR']['predicted_category'].value_counts()
            for category, count in category_counts.items():
                print(f"  - {category}: {count}")
        
        print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main() 