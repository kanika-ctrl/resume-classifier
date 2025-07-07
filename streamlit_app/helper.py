import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_results(prediction, probabilities, model_classes):
    """Display classification results in a nice format."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Predicted Category:** {prediction}")
        confidence = max(probabilities)
        st.info(f"**Confidence:** {confidence:.2%}")
    
    with col2:
        st.write("**Category Probabilities:**")
        # Create a sorted list of (category, probability) tuples
        cat_probs = list(zip(model_classes, probabilities))
        cat_probs.sort(key=lambda x: x[1], reverse=True)
        
        for category, prob in cat_probs:
            if prob > 0.01:  # Show only significant probabilities
                st.write(f"- {category}: {prob:.2%}")

def plot_probabilities(probabilities, model_classes):
    """Create a bar chart of category probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by probability
    sorted_data = sorted(zip(model_classes, probabilities), key=lambda x: x[1], reverse=True)
    categories, probs = zip(*sorted_data)
    
    # Create bar plot
    bars = ax.bar(range(len(categories)), probs)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Probability')
    ax.set_title('Category Probabilities')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def validate_file_upload(uploaded_file):
    """Validate uploaded file format and size."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File too large (max 10MB)"
    
    # Check file type
    allowed_types = ['pdf', 'docx', 'txt']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_types:
        return False, f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
    
    return True, "File valid"

def show_sample_data():
    """Display sample data statistics."""
    try:
        df = pd.read_csv('../Resume/Resume.csv')
        st.write("**Dataset Statistics:**")
        st.write(f"- Total resumes: {len(df)}")
        st.write(f"- Categories: {df['Category'].nunique()}")
        st.write(f"- Categories: {', '.join(df['Category'].unique())}")
    except Exception as e:
        st.warning("Could not load sample data statistics") 