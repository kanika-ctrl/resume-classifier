# Resume Classifier

A simple machine learning project to classify resumes into different job categories based on their content.

## Project Structure

 
## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   - Open `model.ipynb` in Jupyter
   - Run all cells to train and save the model

3. **Run the web app:**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

## Usage

### Training
- The `model.ipynb` notebook contains the complete training pipeline
- It loads the Resume.csv data, cleans it, extracts features, and trains a classifier
- The trained model is saved in `saved_models/`

### Web Interface
- Upload resume files (PDF, DOCX, TXT) or paste text
- Get instant classification results with confidence scores
- View probability distribution across all categories

## Features

- **Text Cleaning:** Removes special characters, normalizes whitespace
- **Feature Extraction:** TF-IDF vectorization with configurable features
- **Multiple File Formats:** Supports PDF, DOCX, and TXT files
- **Web Interface:** User-friendly Streamlit app
- **Model Persistence:** Save and load trained models

## Model Performance

The model uses:
- **Algorithm:** Multinomial Naive Bayes
- **Features:** TF-IDF vectors (1000 features)
- **Text Processing:** Lowercase, special character removal, stopword removal

## File Formats Supported

- **PDF:** Using PyPDF2
- **DOCX:** Using python-docx
- **TXT:** Direct text reading

## Dependencies

- pandas: Data manipulation
- scikit-learn: Machine learning
- streamlit: Web interface
- PyPDF2: PDF parsing
- python-docx: DOCX parsing
- joblib: Model serialization

## Contributing

Feel free to improve the code, add new features, or fix bugs! 
