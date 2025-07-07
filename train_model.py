#!/usr/bin/env python3
"""
Simple script to train the resume classifier model.
Run this instead of the Jupyter notebook if you prefer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

# Add project modules to path
sys.path.append('.')
from data_cleaning.cleaner import clean_resume_data
from feature_extraction.vectorizer import ResumeVectorizer

def main():
    print("🚀 Starting Resume Classifier Training...")
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('Resume/Resume.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean data
    print("🧹 Cleaning data...")
    df_clean = clean_resume_data(df, text_column='Resume_str')
    
    # Prepare features and labels
    X = df_clean['cleaned_text']
    y = df_clean['Category']
    
    print(f"Categories: {y.unique()}")
    print(f"Category distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Feature extraction
    print("🔧 Extracting features...")
    vectorizer = ResumeVectorizer(max_features=1000)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # Train model
    print("🤖 Training model...")
    model = MultinomialNB()
    model.fit(X_train_features, y_train)
    
    # Evaluate
    print("📈 Evaluating model...")
    y_pred = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("💾 Saving model...")
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(model, 'saved_models/resume_classifier.pkl')
    vectorizer.save('saved_models/vectorizer.pkl')
    print("Model saved successfully!")
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    sample_text = "Experienced software developer with Python and Java skills"
    cleaned_sample = clean_resume_data(pd.DataFrame({'Resume': [sample_text]}))['cleaned_text'].iloc[0]
    features = vectorizer.transform([cleaned_sample])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    print(f"Sample text: {sample_text}")
    print(f"Predicted category: {prediction}")
    print(f"Confidence: {max(probability):.4f}")

if __name__ == "__main__":
    main() 