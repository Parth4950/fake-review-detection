

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pickle

# Use relative paths - works from project root directory
TRAINING_DATA_PATH = os.path.join('Predicted Data', 'Training', 'reviews.csv')
GENERATED_REVIEWS_PATH = os.path.join('Generated reviews', 'flipkart_reviews.csv')
MODELS_FOLDER = 'models'

def main():
    """
    Main function to train a model and make predictions on scraped reviews.
    This script uses relative paths and will work from the project root directory.
    """
    
    # Check if training data exists
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"❌ Error: Training data not found at: {TRAINING_DATA_PATH}")
        print("💡 Please ensure the training CSV file exists in 'Predicted Data/Training/' folder")
        return
    
    # Load the training data
    print("📂 Loading training data...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    
    # Prepare data
    df1 = df.iloc[:, [0, 1, 3, 2]].copy()
    df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)
    
    # Perform text vectorization
    print("🔄 Performing TF-IDF vectorization...")
    vectorization = TfidfVectorizer()
    transformed_output = vectorization.fit_transform(df['text_'])
    df2 = pd.DataFrame(transformed_output.toarray(), columns=vectorization.get_feature_names_out())
    df2['Mean'] = df2.mean(axis=1)
    extracted_col = df2["Mean"]
    df3 = df1.join(extracted_col)
    df4 = df3.drop(['text_'], axis=1)
    
    # Prepare features and labels
    x_train = df4[['category', 'rating', 'Mean']]
    y_train = df4[['label']]
    
    # Train the GaussianNB classifier
    print("🎓 Training GaussianNB classifier...")
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    
    # Create models folder if it doesn't exist
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    
    # Save the trained classifier
    model_filename = os.path.join(MODELS_FOLDER, 'gaussian_nb_model.sav')
    vectorizer_filename = os.path.join(MODELS_FOLDER, 'gaussian_nb_vectorizer.sav')
    
    pickle.dump(classifier, open(model_filename, 'wb'))
    pickle.dump(vectorization, open(vectorizer_filename, 'wb'))
    print(f"✅ Model saved to: {model_filename}")
    print(f"✅ Vectorizer saved to: {vectorizer_filename}")
    
    # Check if test data exists
    if not os.path.exists(GENERATED_REVIEWS_PATH):
        print(f"\n⚠️  Warning: Test data not found at: {GENERATED_REVIEWS_PATH}")
        print("💡 Please scrape reviews first using the 'Scrape Reviews' page in the Streamlit app")
        print("   Or place your test CSV file in 'Generated reviews/flipkart_reviews.csv'")
        return
    
    # Load the Flipkart generated reviews
    print(f"\n📂 Loading test data from: {GENERATED_REVIEWS_PATH}")
    flipkart_reviews_df = pd.read_csv(GENERATED_REVIEWS_PATH)
    
    # Check if required columns exist
    if 'Review' not in flipkart_reviews_df.columns and 'text_' not in flipkart_reviews_df.columns:
        print("❌ Error: Test data must contain 'Review' or 'text_' column")
        print(f"   Available columns: {', '.join(flipkart_reviews_df.columns)}")
        return
    
    # Use 'Review' column if available, otherwise 'text_'
    review_column = 'Review' if 'Review' in flipkart_reviews_df.columns else 'text_'
    
    # Perform the necessary preprocessing on the Flipkart generated reviews
    print("🔄 Processing test data...")
    transformed_reviews = vectorization.transform(flipkart_reviews_df[review_column])
    df_flipkart = pd.DataFrame(transformed_reviews.toarray(), columns=vectorization.get_feature_names_out())
    df_flipkart['Mean'] = df_flipkart.mean(axis=1)
    
    # Check if category and rating columns exist, otherwise use defaults
    if 'category' not in flipkart_reviews_df.columns:
        flipkart_reviews_df['category'] = 0  # Default category
    if 'rating' not in flipkart_reviews_df.columns:
        flipkart_reviews_df['rating'] = 5.0  # Default rating
    
    features_flipkart = df_flipkart[['Mean']].copy()
    features_flipkart['category'] = flipkart_reviews_df['category'].values
    features_flipkart['rating'] = flipkart_reviews_df['rating'].values
    
    # Make predictions using the loaded model
    print("🔮 Making predictions...")
    predictions = classifier.predict(features_flipkart)
    
    # Add the predictions to the Flipkart reviews DataFrame
    flipkart_reviews_df['predicted_label'] = predictions
    
    # Save predictions
    output_path = os.path.join('Predicted Data', 'official_predictions.csv')
    os.makedirs('Predicted Data', exist_ok=True)
    flipkart_reviews_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Predictions completed!")
    print(f"📊 Total reviews processed: {len(flipkart_reviews_df)}")
    print(f"💾 Results saved to: {output_path}")
    print("\n📋 Sample predictions:")
    print(flipkart_reviews_df[[review_column, 'predicted_label']].head(10))
    
    # Show statistics
    label_counts = flipkart_reviews_df['predicted_label'].value_counts()
    print(f"\n📈 Prediction distribution:")
    for label, count in label_counts.items():
        print(f"   Label {label}: {count} reviews ({count/len(flipkart_reviews_df)*100:.1f}%)")

if __name__ == "__main__":
    main()

