import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import sys

# Add project modules to path
sys.path.append('..')
from data_cleaning.cleaner import clean_text
from resume_parser.parser import parse_resume_file

# Page config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model_path = '../saved_models/resume_classifier.pkl'
    vectorizer_path = '../saved_models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    else:
        st.error("Model files not found! Please train the model first.")
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

# Main app
def main():
    st.title("📄 Resume Classifier")
    st.write("Upload a resume or paste text to classify it into job categories.")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Options")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Text Input", "File Upload"]
    )
    
    # Main content
    if input_method == "Text Input":
        st.header("Text Input")
        resume_text = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="Enter resume text..."
        )
        
        if st.button("Classify Resume") and resume_text.strip():
            with st.spinner("Classifying..."):
                prediction, probabilities = predict_category(resume_text, model, vectorizer)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Predicted Category:** {prediction}")
                    confidence = max(probabilities)
                    st.info(f"**Confidence:** {confidence:.2%}")
                
                with col2:
                    st.write("**Category Probabilities:**")
                    for i, prob in enumerate(probabilities):
                        if prob > 0.01:  # Show only significant probabilities
                            st.write(f"- {model.classes_[i]}: {prob:.2%}")
    
    else:  # File Upload
        st.header("File Upload")
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx', 'txt']
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Parse file
                resume_text = parse_resume_file(temp_path)
                
                if resume_text:
                    st.text_area("Extracted Text:", resume_text, height=200)
                    
                    if st.button("Classify Resume"):
                        with st.spinner("Classifying..."):
                            prediction, probabilities = predict_category(resume_text, model, vectorizer)
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.success(f"**Predicted Category:** {prediction}")
                                confidence = max(probabilities)
                                st.info(f"**Confidence:** {confidence:.2%}")
                            
                            with col2:
                                st.write("**Category Probabilities:**")
                                for i, prob in enumerate(probabilities):
                                    if prob > 0.01:
                                        st.write(f"- {model.classes_[i]}: {prob:.2%}")
                else:
                    st.error("Could not extract text from the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Resume Classification Model")

if __name__ == "__main__":
    main() 