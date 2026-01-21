import os
import spacy
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import logging
import warnings

# Suppress warnings and logs
warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.ERROR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import re

def get_sentiment(text):
    """Extract sentiment polarity and subjectivity from text"""
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0

def preprocess_text(text):
    """Enhanced preprocessing with TF-IDF ready text cleaning"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def training():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .training-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="training-header">🎓 Model Training</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <h3>📚 Training Information</h3>
    <p>Train machine learning models to detect fake reviews using:</p>
    <ul>
    <li>TF-IDF Vectorization (5000 features, bigrams)</li>
    <li>Sentiment Analysis (Polarity & Subjectivity)</li>
    <li>Multiple ML Algorithms (Random Forest, Logistic Regression, SVM, Naive Bayes)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Use relative paths
    training_folder = os.path.join('Predicted Data', 'Training')
    models_folder = 'models'
    
    # Create models folder if it doesn't exist
    os.makedirs(models_folder, exist_ok=True)
    
    # Check if training folder exists
    if not os.path.exists(training_folder):
        st.error(f"Training folder not found at: {training_folder}")
        st.info("Please ensure training data is available in the 'Predicted Data/Training' folder")
        return
    
    # Get available CSV files
    files = [f for f in os.listdir(training_folder) if f.endswith('.csv')]
    
    # Allow file upload as well
    st.subheader("📂 Training Data Selection")
    
    option = st.radio("Choose data source:", ["Use existing file", "Upload new file"], horizontal=True)
    
    if option == "Upload new file":
        uploaded_file_obj = st.file_uploader("Upload a CSV file", type=['csv'], key="train_upload")
        if uploaded_file_obj is not None:
            # Save uploaded file temporarily
            temp_path = os.path.join(training_folder, uploaded_file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file_obj.getbuffer())
            uploaded_file = uploaded_file_obj.name
            st.success(f"✅ File uploaded: {uploaded_file}")
        else:
            if not files:
                st.error("No CSV files found. Please upload a training file.")
                return
            uploaded_file = files[0] if files else None
    else:
        if not files:
            st.error("No CSV files found in the training folder")
            st.info("💡 Use 'Upload new file' option to add training data")
            return
        
        uploaded_file = st.selectbox('Select a training file', files, key="train_select")
    
    # Create a dropdown menu to select a classification model
    model_choice = st.selectbox("🤖 Choose a classification model",
                                ["Random Forest", "Logistic Regression", "Support Vector Machine", "Multinomial Naive Bayes"])

    # Load spacy model with error handling
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.error("spaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
        return

    def train_model(df, model_choice):
        # Check required columns
        if 'text_' not in df.columns or 'label' not in df.columns:
            st.error("Dataset must contain 'text_' and 'label' columns")
            return
        
        # Preprocess the text data
        with st.spinner('Preprocessing text data and extracting features....'):
            # Clean text
            df['cleaned_text'] = df['text_'].apply(preprocess_text)
            
            # Remove empty texts
            df = df[df['cleaned_text'].str.len() > 0].copy()
            
            if len(df) == 0:
                st.error("No valid text data after preprocessing")
                return
            
            # Extract sentiment features
            st.info("Extracting sentiment features...")
            sentiment_features = df['cleaned_text'].apply(lambda x: pd.Series(get_sentiment(x), index=['polarity', 'subjectivity']))
            df['sentiment_polarity'] = sentiment_features['polarity']
            df['sentiment_subjectivity'] = sentiment_features['subjectivity']
            
            # Convert labels to numeric if needed
            if df['label'].dtype == 'object':
                unique_labels = df['label'].unique()
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                df['label_numeric'] = df['label'].map(label_map)
                st.info(f"Label mapping: {label_map}")
            else:
                df['label_numeric'] = df['label']
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                df[['cleaned_text', 'sentiment_polarity', 'sentiment_subjectivity']], 
                df['label_numeric'], 
                test_size=0.2,
                random_state=42,
                stratify=df['label_numeric']
            )
            
            # Extract sentiment features for train/test
            train_sentiment = X_train[['sentiment_polarity', 'sentiment_subjectivity']].values
            test_sentiment = X_test[['sentiment_polarity', 'sentiment_subjectivity']].values
            
            # Extract text for vectorization
            X_train_text = X_train['cleaned_text']
            X_test_text = X_test['cleaned_text']

        with st.spinner('Applying TF-IDF vectorization and training model....'):
            # Convert text data to numerical features using TfidfVectorizer with optimized parameters
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            
            X_train_vec = vectorizer.fit_transform(X_train_text)
            X_test_vec = vectorizer.transform(X_test_text)
            
            # Combine TF-IDF features with sentiment features
            from scipy.sparse import hstack
            from scipy.sparse import csr_matrix
            import numpy as np
            
            # Convert sentiment arrays to sparse matrices for proper stacking
            train_sentiment_sparse = csr_matrix(train_sentiment)
            test_sentiment_sparse = csr_matrix(test_sentiment)
            
            X_train_combined = hstack([X_train_vec, train_sentiment_sparse])
            X_test_combined = hstack([X_test_vec, test_sentiment_sparse])

            # Train a classification model on the training data
            if model_choice == "Random Forest":
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                # Random Forest can handle sparse matrices
                X_train_final = X_train_combined
                X_test_final = X_test_combined
            elif model_choice == "Logistic Regression":
                clf = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs',
                    multi_class='auto'
                )
                # Logistic Regression can handle sparse matrices
                X_train_final = X_train_combined
                X_test_final = X_test_combined
            elif model_choice == "Support Vector Machine":
                # SVM needs dense matrices when combined with other features
                clf = SVC(kernel='rbf', probability=True, random_state=42, max_iter=1000)
                # Convert to dense for SVM
                X_train_final = X_train_combined.toarray()
                X_test_final = X_test_combined.toarray()
                st.info("ℹ️ Converting to dense matrix for SVM (this may take a moment for large datasets)...")
            elif model_choice == "Multinomial Naive Bayes":
                # MultinomialNB works with sparse matrices, but needs non-negative values
                # TF-IDF can have negative values when combined, so we'll use only TF-IDF part
                # and add sentiment as separate features
                clf = MultinomialNB(alpha=1.0)
                # For MultinomialNB, we'll use TF-IDF only and handle sentiment differently
                # Convert sentiment to absolute values and scale
                train_sentiment_abs = np.abs(train_sentiment)
                test_sentiment_abs = np.abs(test_sentiment)
                # Scale sentiment to positive range for MultinomialNB
                scaler = MinMaxScaler()
                train_sentiment_scaled = scaler.fit_transform(train_sentiment_abs)
                test_sentiment_scaled = scaler.transform(test_sentiment_abs)
                # Combine with TF-IDF
                train_sentiment_sparse_nb = csr_matrix(train_sentiment_scaled)
                test_sentiment_sparse_nb = csr_matrix(test_sentiment_scaled)
                X_train_final = hstack([X_train_vec, train_sentiment_sparse_nb])
                X_test_final = hstack([X_test_vec, test_sentiment_sparse_nb])
                # Save scaler for MultinomialNB
                scaler_filename = os.path.join(models_folder, model_choice.lower().replace(" ", "_") + '_scaler.sav')
                pickle.dump(scaler, open(scaler_filename, 'wb'))
            else:
                st.error(f"Unknown model: {model_choice}")
                return

            clf.fit(X_train_final, y_train)

            # Evaluate the performance of the trained model
            train_score = clf.score(X_train_final, y_train)
            test_score = clf.score(X_test_final, y_test)
            
            y_pred = clf.predict(X_test_final)
            
            # Display results
            st.success("Model training completed!")
            st.metric("Training Accuracy", f"{train_score:.4f}")
            st.metric("Test Accuracy", f"{test_score:.4f}")
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, 
                                     index=[f'True {i}' for i in range(len(cm))],
                                     columns=[f'Pred {i}' for i in range(len(cm))]))

            # Save the trained model as a binary file
            model_filename = os.path.join(models_folder, model_choice.lower().replace(" ", "_") + '_model.sav')
            vectorizer_filename = os.path.join(models_folder, model_choice.lower().replace(" ", "_") + '_vectorizer.sav')
            
            pickle.dump(clf, open(model_filename, 'wb'))
            pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
            
            # Note: Scaler for MultinomialNB is saved above if needed
            
            st.success(f"Model saved successfully!")
            st.info(f"Model: {model_filename}")
            st.info(f"Vectorizer: {vectorizer_filename}")

    # Load and display data info
    if uploaded_file:
        file_path = os.path.join(training_folder, uploaded_file)
        try:
            df = pd.read_csv(file_path)
            st.success(f"✅ Loaded: {uploaded_file}")
            st.info(f"📊 Dataset: {len(df)} rows × {len(df.columns)} columns")
            
            # Show column info
            if 'text_' in df.columns and 'label' in df.columns:
                st.success("✅ Required columns found: 'text_' and 'label'")
            else:
                st.error("❌ Missing required columns!")
                st.info(f"Available columns: {', '.join(df.columns)}")
                st.info("Required: 'text_' (review text) and 'label' (CG/OR or 0/1)")
                return
            
            # Show preview
            with st.expander("📋 Preview Data (Click to expand)"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                train_model(df, model_choice)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Please check if the file is a valid CSV with proper formatting")
