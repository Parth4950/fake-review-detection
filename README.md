# Fake Review Detection System

A machine learning-based system for detecting fake product reviews using TF-IDF vectorization, sentiment analysis, and multiple ML models. Includes a Streamlit web interface for scraping Flipkart reviews, training models, and testing predictions.

## Features

- **Web Scraping**: Scrape product reviews from Flipkart
- **Text Preprocessing**: Advanced text cleaning (removes URLs, emojis, HTML tags)
- **TF-IDF Vectorization**: Convert text to numerical features
- **Sentiment Analysis**: Extract polarity and subjectivity using TextBlob
- **ML Models**: Random Forest, Logistic Regression, SVM, Multinomial Naive Bayes
- **Interactive UI**: Streamlit-based web interface

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fake-Reviews-Detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
   ```

## Usage

### Start the Application
```bash
streamlit run main.py
```

### Workflow

1. **Scrape Reviews** (`🛍️ Scrape Reviews`)
   - Enter Flipkart product URL
   - Select number of pages to scrape
   - Reviews saved to `Generated reviews/flipkart_reviews.csv`

2. **Train Model** (`🎓 Training`)
   - Select or upload training CSV (must have `text_` and `label` columns)
   - Choose model (Random Forest, Logistic Regression, SVM, Naive Bayes)
   - Model saved to `models/` folder

3. **Test Model** (`🧪 Testing`)
   - Select trained model
   - Use scraped reviews or upload test CSV
   - View predictions, statistics, and visualizations

## Training Data Format

CSV file must contain:
- `text_`: Review text (required)
- `label`: 'CG' (fake) or 'OR' (genuine), or 0/1 (required)

Example:
```csv
text_,label
"Great product!",OR
"Fake review",CG
```

## Project Structure

```
Fake-Reviews-Detection/
├── main.py              # Streamlit app entry point
├── home_page.py         # Scraping interface
├── training.py          # Model training
├── testing.py           # Model testing
├── clean_data.py        # Text preprocessing
├── csvgencode.py      # Scraping logic
├── url_modifier.py     # URL utilities
├── models/             # Saved models (auto-created)
├── Generated reviews/  # Scraped reviews (auto-created)
└── Predicted Data/     # Predictions and training data
```

## Dependencies

- streamlit, pandas, numpy
- scikit-learn, scipy
- spacy, textblob
- beautifulsoup4, requests
- wordcloud, matplotlib, altair, plotly

See `requirements.txt` for versions.

## Troubleshooting

- **spaCy model not found**: `python -m spacy download en_core_web_sm`
- **Module errors**: `pip install -r requirements.txt`
- **Scraping fails**: Check URL format and internet connection
- **Training errors**: Ensure CSV has `text_` and `label` columns

## License

Educational purposes only.
