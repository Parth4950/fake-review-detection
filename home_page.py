import streamlit as st
from csvgencode import get_reviews
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import time
import plotly.express as px
import os
import re


def home_page():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #000000;
    }
    .info-box strong {
        color: #000000;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: #155724;
    }
    .product-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🛍️ Flipkart Product Review Scraper</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>📋 Instructions:</strong><br>
    1. Enter a valid Flipkart product URL (e.g., https://www.flipkart.com/product-name/p/...)<br>
    2. Click "Scrape Reviews" to extract product reviews<br>
    3. Reviews will be saved for model training and testing
    </div>
    """, unsafe_allow_html=True)
    
    user_link = st.text_input("🔗 Enter Flipkart Product URL", "", placeholder="https://www.flipkart.com/...")
    
    num_pages = st.slider("📄 Number of pages to scrape", 1, 10, 2, key="pages_slider")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        scrape_button = st.button("🚀 Scrape Reviews", type="primary", use_container_width=True)
    
    if scrape_button:
        if not user_link:
            st.error("❌ Please enter a URL")
            return
        
        # Validate URL
        if not (user_link.startswith("http://") or user_link.startswith("https://")):
            st.error("❌ Invalid URL format. Please include http:// or https://")
            return
        
        if "flipkart.com" not in user_link.lower():
            st.error("❌ Please enter a valid Flipkart product URL")
            return
        
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            with st.spinner('🔄 Fetching product information...'):
                response = requests.get(user_link, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
            
            # Try multiple selectors for product name and reviews count
            product_name = None
            reviews_count = None
            
            # Try different selectors for product name
            name_selectors = [
                ('span', {'class': 'B_NuCI'}),
                ('h1', {'class': 'yhB1nd'}),
                ('span', {'class': 'VU-ZEz'}),
                ('h1', {'class': '_2YhO5f'})
            ]
            
            for tag, attrs in name_selectors:
                element = soup.find(tag, attrs)
                if element:
                    product_name = element.text.strip()
                    break
            
            # Try different selectors for reviews count
            count_selectors = [
                ('span', {'class': '_2_R_DZ'}),
                ('span', {'class': '_1YokD2'}),
                ('div', {'class': '_3UAT2v'})
            ]
            
            for tag, attrs in count_selectors:
                element = soup.find(tag, attrs)
                if element:
                    text = element.text.strip()
                    # Extract number from text
                    numbers = re.findall(r'\d+', text.replace(',', ''))
                    if numbers:
                        reviews_count = numbers[0]
                        break
            
            if product_name:
                st.markdown(f"""
                <div class="product-card">
                <h2>📦 {product_name}</h2>
                <p>📊 Total Reviews: {reviews_count if reviews_count else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not extract product name, but continuing with scraping...")
            
            # Scrape reviews
            with st.spinner('🔄 Scraping reviews (this may take a minute)...'):
                reviews = get_reviews(1, num_pages, user_link)
                
                if not reviews:
                    st.error("❌ No reviews found. The product might not have reviews or the URL structure has changed.")
                    st.info("💡 Try a different product URL or check if the product has reviews on Flipkart.")
                    return
                
                # Create directory if it doesn't exist
                os.makedirs("Generated reviews", exist_ok=True)
                
                # Save reviews
                output_file = "Generated reviews/flipkart_reviews.csv"
                with open(output_file, mode="w", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["Rating", "Review"])
                    for review, rating in reviews:
                        writer.writerow([rating, review])
                
                # Display success message
                st.markdown(f"""
                <div class="success-box">
                <strong>✅ Scraping completed successfully!</strong><br>
                📝 Found {len(reviews)} reviews<br>
                💾 Saved to: {output_file}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sample reviews
                if reviews:
                    st.subheader("📋 Sample Reviews")
                    sample_df = pd.DataFrame(reviews[:5], columns=["Rating", "Review"])
                    st.dataframe(sample_df, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(reviews))
                    with col2:
                        avg_rating = sum([float(r[0]) for r in reviews if r[0].replace('.','').isdigit()]) / len([r for r in reviews if r[0].replace('.','').isdigit()]) if reviews else 0
                        st.metric("Average Rating", f"{avg_rating:.2f}")
                    with col3:
                        st.metric("Status", "✅ Ready")
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")
            st.info("💡 Please check your internet connection and try again.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 The URL might be invalid or Flipkart's page structure has changed. Try a different product URL.")
            st.code(f"Error details: {type(e).__name__}")


            # Save reviews to a CSV file

            # positive negative average reviews
            # df = pd.read_csv("C:/Users/Akshay/Desktop/flipkart reviews/Generated reviews/flipkart_reviews.csv")
            # Load data
