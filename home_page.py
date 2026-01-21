"""
Home Page - Flipkart Review Scraper UI
Clean, production-quality Streamlit interface with proper error handling
"""

import streamlit as st
from csvgencode import get_reviews
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import os
import re
import logging

# Suppress all unnecessary logs and warnings
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.ERROR)


def home_page():
    """Main home page function with clean UI and error handling."""
    
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
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        color: #721c24;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
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
    2. Select the number of review pages to scrape<br>
    3. Click "Scrape Reviews" to extract product reviews<br>
    4. Reviews will be saved and ready for model training and testing
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.subheader("📥 Input")
    user_link = st.text_input("🔗 Enter Flipkart Product URL", "", placeholder="https://www.flipkart.com/...")
    
    num_pages = st.slider("📄 Number of pages to scrape", 1, 10, 2, key="pages_slider")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        scrape_button = st.button("🚀 Scrape Reviews", type="primary", use_container_width=True)
    
    if scrape_button:
        # Validate input
        if not user_link:
            st.markdown("""
            <div class="error-box">
            <strong>❌ Please enter a Flipkart product URL</strong>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Validate URL format
        if not (user_link.startswith("http://") or user_link.startswith("https://")):
            st.markdown("""
            <div class="error-box">
            <strong>❌ Invalid URL format</strong><br>
            Please include http:// or https:// in the URL
            </div>
            """, unsafe_allow_html=True)
            return
        
        if "flipkart.com" not in user_link.lower():
            st.markdown("""
            <div class="error-box">
            <strong>❌ Invalid URL</strong><br>
            Please enter a valid Flipkart product URL
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Fetch product information
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            with st.spinner('🔄 Fetching product information...'):
                response = requests.get(user_link, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract product name
            product_name = None
            name_selectors = [
                ('span', {'class': 'B_NuCI'}),
                ('h1', {'class': 'yhB1nd'}),
                ('span', {'class': 'VU-ZEz'}),
                ('h1', {'class': '_2YhO5f'})
            ]
            
            for tag, attrs in name_selectors:
                try:
                    element = soup.find(tag, attrs)
                    if element:
                        product_name = element.text.strip()
                        break
                except:
                    continue
            
            # Display product info
            if product_name:
                st.markdown(f"""
                <div class="product-card">
                <h2>📦 {product_name}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        except requests.exceptions.Timeout:
            st.markdown("""
            <div class="error-box">
            <strong>❌ Connection Timeout</strong><br>
            The request took too long. Please check your internet connection and try again.
            </div>
            """, unsafe_allow_html=True)
            return
        except requests.exceptions.RequestException:
            st.markdown("""
            <div class="error-box">
            <strong>❌ Network Error</strong><br>
            Could not connect to Flipkart. Please check your internet connection and try again.
            </div>
            """, unsafe_allow_html=True)
            return
        except Exception:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Could not extract product name</strong><br>
            Continuing with scraping anyway...
            </div>
            """, unsafe_allow_html=True)
        
        # Scrape reviews
        try:
            with st.spinner('🔄 Scraping reviews (this may take a minute)...'):
                reviews = get_reviews(1, num_pages, user_link)
            
            # Process results
            if not reviews:
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ No reviews found</strong><br>
                The product might not have reviews, or Flipkart's page structure has changed.<br>
                Please try a different product URL or check if the product has reviews on Flipkart.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Create output directory
            os.makedirs("Generated reviews", exist_ok=True)
            
            # Save reviews to CSV
            output_file = "Generated reviews/flipkart_reviews.csv"
            try:
                with open(output_file, mode="w", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["Rating", "Review", "Username", "Date"])
                    for rating, review_text, username, date in reviews:
                        writer.writerow([rating, review_text, username, date])
                
                # Display success message
                st.markdown(f"""
                <div class="success-box">
                <strong>✅ Scraping completed successfully!</strong><br>
                📝 Found {len(reviews)} reviews<br>
                💾 Saved to: {output_file}
                </div>
                """, unsafe_allow_html=True)
            
            except Exception:
                st.markdown("""
                <div class="error-box">
                <strong>❌ Error saving reviews</strong><br>
                Could not write to file. Please check file permissions.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Display sample reviews
            st.subheader("📋 Sample Reviews")
            sample_data = []
            for rating, review_text, username, date in reviews[:10]:
                sample_data.append({
                    "Rating": rating,
                    "Review": review_text[:200] + "..." if len(review_text) > 200 else review_text,
                    "Username": username if username else "N/A",
                    "Date": date if date else "N/A"
                })
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
            
            # Display statistics
            st.subheader("📊 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reviews", len(reviews))
            
            with col2:
                # Calculate average rating
                try:
                    numeric_ratings = [float(r[0]) for r in reviews if r[0] and r[0].replace('.','').isdigit()]
                    if numeric_ratings:
                        avg_rating = sum(numeric_ratings) / len(numeric_ratings)
                        st.metric("Average Rating", f"{avg_rating:.2f}")
                    else:
                        st.metric("Average Rating", "N/A")
                except:
                    st.metric("Average Rating", "N/A")
            
            with col3:
                # Count reviews with usernames
                usernames_count = sum(1 for r in reviews if r[2] and r[2].strip())
                st.metric("With Username", usernames_count)
            
            with col4:
                st.metric("Status", "✅ Ready")
            
            # Rating distribution chart
            if len(reviews) > 0:
                try:
                    rating_counts = {}
                    for rating, _, _, _ in reviews:
                        if rating and rating.isdigit() and 1 <= int(rating) <= 5:
                            rating_counts[rating] = rating_counts.get(rating, 0) + 1
                    
                    if rating_counts:
                        st.subheader("📈 Rating Distribution")
                        rating_df = pd.DataFrame(list(rating_counts.items()), columns=["Rating", "Count"])
                        rating_df = rating_df.sort_values("Rating")
                        st.bar_chart(rating_df.set_index("Rating"))
                except:
                    # Silently skip chart if there's an error
                    pass
        
        except Exception:
            # Generic error handler - show user-friendly message
            st.markdown("""
            <div class="error-box">
            <strong>❌ Scraping Error</strong><br>
            An error occurred while scraping reviews. This could be due to:<br>
            • Flipkart blocking automated requests (CAPTCHA)<br>
            • Changes in Flipkart's page structure<br>
            • Network connectivity issues<br>
            • The product URL is invalid<br><br>
            Please try again with a different product URL or check your internet connection.
            </div>
            """, unsafe_allow_html=True)
