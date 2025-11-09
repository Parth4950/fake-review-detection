import os
import pandas as pd
import pickle
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.sparse import hstack
from textblob import TextBlob
import re

# Define the preprocessing function for text data
from clean_data import clean_text
from training import preprocess_text, get_sentiment

def test_model():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .testing-header {
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="testing-header">🧪 Model Testing</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <h3>🔬 Test Your Model</h3>
    <p>Select a trained model and test it on your scraped reviews to detect fake reviews.</p>
    </div>
    """, unsafe_allow_html=True)
    
    models_folder = 'models'
    generated_reviews_folder = 'Generated reviews'
    predicted_data_folder = 'Predicted Data'
    
    # Create folders if they don't exist
    os.makedirs(generated_reviews_folder, exist_ok=True)
    os.makedirs(predicted_data_folder, exist_ok=True)
    
    # Check for available models
    if not os.path.exists(models_folder):
        st.error(f"Models folder not found at: {models_folder}")
        st.info("Please train a model first using the Training page")
        return
    
    model_files = [f for f in os.listdir(models_folder) if f.endswith('_model.sav')]
    
    if not model_files:
        st.error("No trained models found. Please train a model first.")
        return
    
    # Let user select a model
    selected_model_file = st.selectbox('Select a trained model', model_files)
    
    # Extract model name to find corresponding vectorizer
    model_name = selected_model_file.replace('_model.sav', '')
    vectorizer_file = os.path.join(models_folder, model_name + '_vectorizer.sav')
    model_file = os.path.join(models_folder, selected_model_file)
    
    if not os.path.exists(vectorizer_file):
        st.error(f"Vectorizer file not found: {vectorizer_file}")
        return
    
    # Load model and vectorizer
    try:
        with st.spinner('Loading model and vectorizer...'):
            loaded_model = pickle.load(open(model_file, 'rb'))
            vectorizer = pickle.load(open(vectorizer_file, 'rb'))
        st.success("Model and vectorizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Check for test data
    test_data_path = os.path.join(generated_reviews_folder, 'flipkart_reviews.csv')
    
    # Allow file upload option
    st.subheader("📂 Test Data Selection")
    use_uploaded = st.checkbox("Upload a different CSV file", value=False)
    
    if use_uploaded:
        uploaded_file = st.file_uploader("Upload CSV file with reviews", type=['csv'], key="test_upload")
        if uploaded_file is not None:
            try:
                df_unlabeled = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
        else:
            if os.path.exists(test_data_path):
                st.info("💡 Using default scraped reviews file")
                df_unlabeled = pd.read_csv(test_data_path)
            else:
                st.warning("Please upload a file or scrape reviews first")
                return
    else:
        if not os.path.exists(test_data_path):
            st.warning(f"Test data not found at: {test_data_path}")
            st.info("💡 Please scrape reviews first or check 'Upload a different CSV file'")
            return
        df_unlabeled = pd.read_csv(test_data_path)
        st.info(f"📂 Using scraped reviews: {test_data_path}")
    
    # Show data info
    st.info(f"📊 Loaded {len(df_unlabeled)} reviews with columns: {', '.join(df_unlabeled.columns)}")
    
    # Check required columns
    review_column = None
    for col in ['Review', 'review', 'text', 'text_', 'review_text']:
        if col in df_unlabeled.columns:
            review_column = col
            break
    
    if review_column is None:
        st.error("❌ Could not find review column. Expected: 'Review', 'review', 'text', 'text_', or 'review_text'")
        st.info(f"Available columns: {', '.join(df_unlabeled.columns)}")
        st.info("💡 Please ensure your CSV has a column with review text")
        return
    
    st.success(f"✅ Found review column: '{review_column}'")
    
    # Clean Rating column - convert to numeric if possible
    if 'Rating' in df_unlabeled.columns:
        # Check if Rating column has text or numbers
        sample_rating = str(df_unlabeled['Rating'].iloc[0]) if len(df_unlabeled) > 0 else ""
        if len(sample_rating) > 10:  # Likely text, not a number
            st.warning("⚠️ Rating column contains text instead of numbers. This column will be ignored for statistics.")
            # Don't try to convert, just skip it
            df_unlabeled = df_unlabeled.drop(columns=['Rating'], errors='ignore')
        else:
            # Try to convert Rating to numeric, handling errors
            df_unlabeled['Rating'] = pd.to_numeric(df_unlabeled['Rating'], errors='coerce')
            # Fill NaN values with 0 or remove rows with invalid ratings
            invalid_ratings = df_unlabeled['Rating'].isna().sum()
            if invalid_ratings > 0:
                st.warning(f"⚠️ Found {invalid_ratings} reviews with invalid ratings. These will be set to 0.")
                df_unlabeled['Rating'] = df_unlabeled['Rating'].fillna(0)
    
    # Preprocess the unlabelled data
    with st.spinner('Preprocessing text data and extracting sentiment features...'):
        df_unlabeled['cleaned_text'] = df_unlabeled[review_column].apply(clean_text)
        df_unlabeled['cleaned_text'] = df_unlabeled['cleaned_text'].apply(preprocess_text)
        
        # Remove empty texts
        df_unlabeled = df_unlabeled[df_unlabeled['cleaned_text'].str.len() > 0].copy()
        
        if len(df_unlabeled) == 0:
            st.error("No valid text data after preprocessing")
            return
        
        # Extract sentiment features
        sentiment_features = df_unlabeled['cleaned_text'].apply(
            lambda x: pd.Series(get_sentiment(x), index=['polarity', 'subjectivity'])
        )
        df_unlabeled['sentiment_polarity'] = sentiment_features['polarity']
        df_unlabeled['sentiment_subjectivity'] = sentiment_features['subjectivity']
        
        # Convert text to TF-IDF features
        X_unlabeled_vec = vectorizer.transform(df_unlabeled['cleaned_text'])
        
        # Combine TF-IDF with sentiment features
        from scipy.sparse import csr_matrix
        sentiment_array = df_unlabeled[['sentiment_polarity', 'sentiment_subjectivity']].values
        sentiment_sparse = csr_matrix(sentiment_array)
        X_unlabeled_combined = hstack([X_unlabeled_vec, sentiment_sparse])
    
    # Make predictions - handle different model types
    with st.spinner('Making predictions...'):
        try:
            # Check if model needs special handling
            model_type = type(loaded_model).__name__
            
            # Handle MultinomialNB - check if scaler exists
            if model_type == 'MultinomialNB':
                scaler_file = os.path.join(models_folder, model_name + '_scaler.sav')
                if os.path.exists(scaler_file):
                    scaler = pickle.load(open(scaler_file, 'rb'))
                    # Scale sentiment features for MultinomialNB
                    sentiment_abs = np.abs(df_unlabeled[['sentiment_polarity', 'sentiment_subjectivity']].values)
                    sentiment_scaled = scaler.transform(sentiment_abs)
                    from scipy.sparse import csr_matrix
                    sentiment_sparse_nb = csr_matrix(sentiment_scaled)
                    X_unlabeled_final = hstack([X_unlabeled_vec, sentiment_sparse_nb])
                else:
                    X_unlabeled_final = X_unlabeled_combined
            elif model_type == 'SVC':
                # SVM needs dense matrix
                X_unlabeled_final = X_unlabeled_combined.toarray()
            else:
                # Other models can handle sparse
                X_unlabeled_final = X_unlabeled_combined
            
            predicted_labels = loaded_model.predict(X_unlabeled_final)
            
            # Get prediction probabilities if available
            try:
                predicted_probs = loaded_model.predict_proba(X_unlabeled_final)
                df_unlabeled['prediction_confidence'] = np.max(predicted_probs, axis=1)
            except:
                df_unlabeled['prediction_confidence'] = None
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
            st.info("💡 The model might be incompatible. Try retraining with a different model.")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Add predicted labels
    df_unlabeled['predicted_label'] = predicted_labels
    
    # Save predictions
    output_path = os.path.join(predicted_data_folder, 'predicted_data.csv')
    df_unlabeled.to_csv(output_path, index=False)
    st.success(f"Predictions saved to: {output_path}")
    
    # Display statistics
    st.subheader("📊 Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df_unlabeled))
    with col2:
        if 'Rating' in df_unlabeled.columns:
            # Safely calculate average rating
            try:
                numeric_ratings = pd.to_numeric(df_unlabeled['Rating'], errors='coerce')
                valid_ratings = numeric_ratings.dropna()
                if len(valid_ratings) > 0:
                    avg_rating = round(valid_ratings.mean(), 1)
                    st.metric("Average Rating", avg_rating)
                else:
                    st.metric("Average Rating", "N/A")
            except:
                st.metric("Average Rating", "N/A")
        else:
            st.metric("Status", "✅ Complete")
    with col3:
        # Count fake reviews (label 0 or 'CG')
        fake_labels = [0, '0', 'CG', 'cg']
        fake_count = df_unlabeled['predicted_label'].isin(fake_labels).sum()
        st.metric("Fake Detected", fake_count)
    
    # Label distribution chart
    label_counts = df_unlabeled["predicted_label"].value_counts()
    label_df = label_counts.reset_index()
    label_df.columns = ['Label', 'Count']
    
    # Map numeric labels to text
    label_map = {0: 'Fake (0)', 1: 'Genuine (1)', 'CG': 'Fake (CG)', 'OR': 'Genuine (OR)'}
    label_df['Label_Text'] = label_df['Label'].map(label_map).fillna(f"Label {label_df['Label']}")
    
    chart = alt.Chart(label_df).mark_bar().encode(
        x=alt.X('Label_Text', axis=alt.Axis(title='Prediction')),
        y=alt.Y('Count', axis=alt.Axis(title='Number of Reviews')),
        color='Label_Text',
        tooltip=['Label_Text', 'Count']
    ).properties(
        width=600,
        height=400,
        title='Fake vs Genuine Reviews Distribution'
    ).interactive()
    st.altair_chart(chart)
    
    # Display sample predictions
    st.subheader("📋 Sample Predictions")
    display_cols = [review_column, 'predicted_label']
    if 'Rating' in df_unlabeled.columns:
        display_cols.insert(1, 'Rating')
    if 'prediction_confidence' in df_unlabeled.columns and df_unlabeled['prediction_confidence'].notna().any():
        display_cols.append('prediction_confidence')
    
    st.dataframe(df_unlabeled[display_cols].head(20), use_container_width=True)
    
    # Word cloud
    if len(df_unlabeled) > 0:
        st.subheader("☁️ Word Cloud")
        text = " ".join(review for review in df_unlabeled['cleaned_text'].astype(str))
        
        if text.strip():
            try:
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
                plt.close()
            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")





