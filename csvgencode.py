import streamlit as st
import csv
import time

from url_modifier import url_modifier
import requests
from bs4 import BeautifulSoup


# Flipkart Review scrapping
def get_reviews(start_page: int, end_page: int, url: str) -> list[tuple[str, str]]:
    import requests
    
    # Clean the input URL
    original_url = url.strip().rstrip('/')
    if not original_url.startswith('http'):
        original_url = 'https://' + original_url
    
    reviews = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://www.flipkart.com/'
    }
    
    # Strategy 1: Try scraping from product page directly (most reliable)
    st.info("🔍 Strategy 1: Scraping reviews from product page...")
    try:
        response = requests.get(original_url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            page_reviews = extract_reviews_from_page(soup, headers)
            if page_reviews:
                reviews.extend(page_reviews)
                st.success(f"✅ Found {len(page_reviews)} reviews on product page!")
    except Exception as e:
        st.warning(f"⚠️ Could not scrape from product page: {str(e)}")
    
    # Strategy 2: Try reviews URL if we have few reviews
    if len(reviews) < 10 and end_page > 1:
        st.info("🔍 Strategy 2: Trying reviews page URL...")
        url_f = url_modifier(original_url)
        if url_f:
            for page in range(start_page, end_page + 1):
                page_url = f"{url_f}{page}"
                try:
                    response = requests.get(page_url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, "html.parser")
                        page_reviews = extract_reviews_from_page(soup, headers)
                        if page_reviews:
                            reviews.extend(page_reviews)
                    elif response.status_code == 404:
                        if page == start_page:
                            st.warning(f"⚠️ Reviews page returned 404. Trying alternative methods...")
                        break
                except:
                    continue
    
    # Strategy 3: Try alternative URL formats
    if not reviews:
        st.info("🔍 Strategy 3: Trying alternative URL formats...")
        # Extract product ID and try different formats
        import re
        product_match = re.search(r'/p/([^/?]+)', original_url)
        if product_match:
            product_id = product_match.group(1).split('?')[0]
            base = 'https://www.flipkart.com'
            alternative_urls = [
                f"{base}/{product_id}/product-reviews",
                f"{base}/reviews/{product_id}",
                f"{original_url}?reviews=true"
            ]
            for alt_url in alternative_urls:
                try:
                    response = requests.get(alt_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, "html.parser")
                        page_reviews = extract_reviews_from_page(soup, headers)
                        if page_reviews:
                            reviews.extend(page_reviews)
                            break
                except:
                    continue
    
    return reviews

def extract_reviews_from_page(soup, headers):
    """Extract reviews from a BeautifulSoup parsed page"""
    reviews = []
    import re
    
    # Check for 404 or error pages
    page_text = soup.get_text().lower()
    if any(phrase in page_text for phrase in ['page not found', '404', 'unfortunately the page']):
        return []
    
    # Try multiple selectors for reviews - updated for current Flipkart structure
    review_selectors = [
        "._2-N8zT",  # Current Flipkart review text selector (2024)
        ".t-ZTKy",  # Alternative selector
        "._2wzgFH",  # Review text alternative
        "div._27M-vq",  # Review container
        "div[class*='_2-N8zT']",  # Partial class match
        "div[class*='t-ZTKy']",  # Partial class match
        "div[class*='review']",  # Generic review class
        "span[class*='review']",  # Generic review span
        "div[data-id*='review']",  # Data attribute
        "._1YokD2._2MImSr",  # Review section
        "div._1AtVbE",  # Review wrapper
        "div._6K-7Co",  # Another review container
        "span._2-N8zT",  # Review text span
    ]
    
    rating_selectors = [
        "._3LWZlK",  # Current Flipkart rating selector (2024)
        "._1BLPMq",  # Alternative rating selector
        "._1VpSqZ",  # Rating alternative
        "div[class*='_3LWZlK']",  # Partial class match
        "div[class*='rating']",  # Generic rating class
        "span[class*='rating']",  # Generic rating span
        "div[title*='star']",  # Star rating in title
        "span[title*='star']",  # Star rating span
    ]
    
    review_containers = []
    rating_containers = []
    
    # Try all selectors and collect all matches (avoid duplicates)
    seen_reviews = set()
    for selector in review_selectors:
        try:
            containers = soup.select(selector)
            for container in containers:
                text = container.get_text(strip=True) if hasattr(container, 'get_text') else str(container).strip()
                if text and text not in seen_reviews and len(text) > 10:
                    review_containers.append(container)
                    seen_reviews.add(text)
        except:
            continue
    
    seen_ratings = set()
    for selector in rating_selectors:
        try:
            containers = soup.select(selector)
            for container in containers:
                text = container.get_text(strip=True) if hasattr(container, 'get_text') else str(container).strip()
                if text and text not in seen_ratings:
                    rating_containers.append(container)
                    seen_ratings.add(text)
        except:
            continue
    
    # Also try finding reviews by text content patterns (fallback method)
    if not review_containers:
        # Look for divs containing review-like text
        all_divs = soup.find_all(['div', 'span', 'p'], class_=True)
        for elem in all_divs:
            text = elem.get_text(strip=True)
            if len(text) > 20 and len(text) < 1000:  # Reasonable review length
                # Check if it looks like a review (has common review words)
                review_keywords = ['good', 'bad', 'product', 'quality', 'nice', 'excellent', 'poor', 
                                 'worth', 'recommend', 'buy', 'delivery', 'packaging', 'value', 'price']
                if any(word in text.lower() for word in review_keywords):
                    # Avoid navigation and UI text
                    if not any(skip in text.lower() for skip in ['menu', 'cart', 'login', 'sign up', 'filter', 'sort']):
                        if text not in seen_reviews:
                            review_containers.append(elem)
                            seen_reviews.add(text)
    
    # Match reviews with ratings
    for i, review_elem in enumerate(review_containers):
        review_text = review_elem.get_text(strip=True) if hasattr(review_elem, 'get_text') else str(review_elem).strip()
        
        # Clean up review text
        review_text = re.sub(r'\s+', ' ', review_text).strip()
        
        if not review_text or len(review_text) < 10:  # Skip very short reviews
            continue
        
        # Skip if it's just a rating number
        if review_text.isdigit() and len(review_text) == 1:
            continue
        
        # Try to find corresponding rating
        rating = "5"  # Default rating
        
        # Look for rating in nearby elements
        if i < len(rating_containers):
            rating_text = rating_containers[i].get_text(strip=True) if hasattr(rating_containers[i], 'get_text') else str(rating_containers[i]).strip()
            rating_match = re.search(r'(\d+)', rating_text)
            if rating_match:
                rating = rating_match.group(1)
        
        # Also check if rating is in the review container's parent
        try:
            parent = review_elem.parent if hasattr(review_elem, 'parent') else None
            if parent:
                parent_text = parent.get_text() if hasattr(parent, 'get_text') else str(parent)
                rating_match = re.search(r'(\d)\s*(?:star|★|⭐)', parent_text, re.IGNORECASE)
                if rating_match:
                    rating = rating_match.group(1)
        except:
            pass
        
        # Avoid duplicates
        if (rating, review_text) not in reviews:
            reviews.append((rating, review_text))
    
    return reviews
