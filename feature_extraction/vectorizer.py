from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

class ResumeVectorizer:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """Fit the vectorizer and transform texts to TF-IDF features."""
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names (words/phrases)."""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save the fitted vectorizer."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, filepath):
        """Load a fitted vectorizer."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True 