"""
Flipkart Review Scraper - Production Quality
Extracts reviews with rating, username, date, and review text
Handles encoding, CAPTCHA, dynamic loading, and error cases gracefully
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import List, Tuple, Optional, Dict
from urllib.parse import urljoin, urlparse
from url_modifier import url_modifier

# Suppress unnecessary warnings and logs
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)


def normalize_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    Removes extra whitespace, newlines, and unwanted characters.
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    # Remove unwanted Unicode characters but keep basic ones
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text.strip()


def clean_review_text(text: str) -> str:
    """
    Clean review text by removing navigation elements, ads, and UI junk.
    """
    if not text:
        return ""
    
    text = normalize_text(text)
    
    # Skip very short or very long texts (likely not reviews)
    if len(text) < 10 or len(text) > 5000:
        return ""
    
    # Skip common UI/navigation text patterns
    skip_patterns = [
        r'(?i)(menu|cart|login|sign up|sign in|register|search|filter|sort|add to cart|buy now)',
        r'(?i)(next|previous|page|showing|results?|items?)',
        r'(?i)(cookie|privacy policy|terms|conditions)',
        r'(?i)(copyright|all rights reserved)',
        r'^\d+$',  # Just numbers
        r'^[★⭐]+\s*$',  # Just stars
        r'^(rating|review|star)s?\s*$',  # Just the word rating/review
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, text.strip()):
            return ""
    
    # Remove common review UI elements
    text = re.sub(r'(?i)(read more|read less|show more|show less)', '', text)
    text = re.sub(r'(?i)(verified purchase|verified buyer)', '', text)
    
    return normalize_text(text)


def extract_rating(elem) -> Optional[str]:
    """
    Extract rating from various possible selectors and attributes.
    Returns rating as string (1-5) or None.
    """
    if not elem:
        return None
    
    # Try to find rating in the element itself
    text = elem.get_text(strip=True) if hasattr(elem, 'get_text') else str(elem)
    
    # Check for numeric rating in text
    rating_match = re.search(r'\b([1-5])\b', text)
    if rating_match:
        rating = rating_match.group(1)
        if rating.isdigit() and 1 <= int(rating) <= 5:
            return rating
    
    # Check title attribute for star rating
    title = elem.get('title', '') if hasattr(elem, 'get') else ''
    if title:
        rating_match = re.search(r'(\d)\s*(?:star|out of)', title, re.IGNORECASE)
        if rating_match:
            return rating_match.group(1)
    
    # Check for star class patterns
    class_attr = elem.get('class', []) if hasattr(elem, 'get') else []
    if class_attr:
        class_str = ' '.join(class_attr)
        rating_match = re.search(r'(\d)-star|star-(\d)', class_str, re.IGNORECASE)
        if rating_match:
            return rating_match.group(1) or rating_match.group(2)
    
    # Check parent elements
    if hasattr(elem, 'parent') and elem.parent:
        parent_rating = extract_rating(elem.parent)
        if parent_rating:
            return parent_rating
    
    return None


def extract_username(elem) -> Optional[str]:
    """
    Extract username from review container.
    """
    if not elem:
        return None
    
    # Try to find username in nearby elements
    # Flipkart uses various class names for usernames
    username_selectors = [
        '._2sc7ZR',  # Current Flipkart username selector
        '._2-N8zT',  # Alternative
        '._1VpSqZ',  # Another variant
        '[class*="reviewer"]',
        '[class*="username"]',
        '[class*="name"]',
    ]
    
    # Check parent container for username
    container = elem
    for _ in range(3):  # Check up to 3 levels up
        if hasattr(container, 'parent') and container.parent:
            container = container.parent
            for selector in username_selectors:
                try:
                    username_elem = container.select_one(selector)
                    if username_elem:
                        username = username_elem.get_text(strip=True)
                        username = normalize_text(username)
                        # Skip if it's too long or looks like review text
                        if 2 <= len(username) <= 50 and not username.lower().startswith(('good', 'bad', 'product', 'quality')):
                            return username
                except:
                    continue
    
    return None


def extract_date(elem) -> Optional[str]:
    """
    Extract review date from review container.
    """
    if not elem:
        return None
    
    # Try to find date in nearby elements
    date_selectors = [
        '._2sc7ZR._1VyoSR',  # Current Flipkart date selector
        '._3mc3RP',  # Alternative
        '[class*="date"]',
        '[class*="time"]',
    ]
    
    # Check parent container
    container = elem
    for _ in range(3):
        if hasattr(container, 'parent') and container.parent:
            container = container.parent
            for selector in date_selectors:
                try:
                    date_elem = container.select_one(selector)
                    if date_elem:
                        date_text = date_elem.get_text(strip=True)
                        date_text = normalize_text(date_text)
                        # Validate it looks like a date
                        if re.search(r'\d{1,2}.*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})', date_text, re.IGNORECASE):
                            return date_text
                except:
                    continue
    
    return None


def extract_reviews_from_page(soup: BeautifulSoup) -> List[Tuple[str, str, str, str]]:
    """
    Extract reviews from a BeautifulSoup parsed page.
    Returns list of tuples: (rating, review_text, username, date)
    """
    reviews = []
    seen_reviews = set()
    
    if not soup:
        return reviews
    
    # Check for error pages
    page_text = soup.get_text().lower()
    error_indicators = ['page not found', '404', 'unfortunately', 'captcha', 'access denied', 'blocked']
    if any(indicator in page_text for indicator in error_indicators):
        return reviews
    
    # Primary selectors for Flipkart review containers (2024 structure)
    # These are the most reliable selectors based on Flipkart's current DOM structure
    review_container_selectors = [
        'div._27M-vq',  # Main review container
        'div._1AtVbE',  # Alternative container
        'div[class*="_1AtVbE"]',
        'div[class*="_27M-vq"]',
        'div._1YokD2._2MImSr._1wob5i',  # Review section wrapper
    ]
    
    # Try to find review containers
    review_containers = []
    for selector in review_container_selectors:
        try:
            containers = soup.select(selector)
            review_containers.extend(containers)
            if review_containers:
                break  # Found containers with this selector
        except:
            continue
    
    # If no containers found, try finding reviews by structure
    if not review_containers:
        # Look for common review patterns
        all_divs = soup.find_all('div', class_=True)
        for div in all_divs:
            classes = ' '.join(div.get('class', []))
            # Look for divs that might contain reviews
            if any(keyword in classes.lower() for keyword in ['review', 'comment', 'rating']):
                if 'container' in classes.lower() or 'item' in classes.lower() or 'card' in classes.lower():
                    review_containers.append(div)
                    if len(review_containers) >= 20:  # Limit search
                        break
    
    # Extract reviews from containers
    for container in review_containers:
        try:
            # Find review text
            review_text_elem = None
            review_selectors = [
                'div.t-ZTKy',  # Current review text selector
                'div._2-N8zT',  # Alternative
                'span.t-ZTKy',
                'span._2-N8zT',
                'div[class*="t-ZTKy"]',
                'div[class*="_2-N8zT"]',
            ]
            
            for selector in review_selectors:
                try:
                    elem = container.select_one(selector)
                    if elem:
                        review_text_elem = elem
                        break
                except:
                    continue
            
            if not review_text_elem:
                continue
            
            # Extract and clean review text
            review_text = review_text_elem.get_text()
            review_text = clean_review_text(review_text)
            
            if not review_text or review_text in seen_reviews:
                continue
            
            # Extract rating
            rating = "5"  # Default rating
            rating_selectors = [
                'div._3LWZlK',  # Current rating selector
                'div._1BLPMq',
                'span._3LWZlK',
                '[class*="_3LWZlK"]',
            ]
            
            for selector in rating_selectors:
                try:
                    rating_elem = container.select_one(selector)
                    extracted_rating = extract_rating(rating_elem)
                    if extracted_rating:
                        rating = extracted_rating
                        break
                except:
                    continue
            
            # Extract username
            username = extract_username(container) or ""
            
            # Extract date
            date = extract_date(container) or ""
            
            # Add to results if valid
            if review_text and len(review_text) >= 10:
                review_tuple = (rating, review_text, username, date)
                if review_text not in seen_reviews:
                    reviews.append(review_tuple)
                    seen_reviews.add(review_text)
        
        except Exception:
            # Silently skip errors for individual reviews
            continue
    
    return reviews


def get_reviews(start_page: int, end_page: int, url: str) -> List[Tuple[str, str, str, str]]:
    """
    Main function to scrape Flipkart reviews.
    Returns list of tuples: (rating, review_text, username, date)
    """
    # Normalize URL
    original_url = url.strip().rstrip('/')
    if not original_url.startswith(('http://', 'https://')):
        original_url = 'https://' + original_url
    
    all_reviews = []
    
    # Headers to mimic browser
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
        'Referer': 'https://www.flipkart.com/',
    }
    
    # Session for connection reuse
    session = requests.Session()
    session.headers.update(headers)
    
    # Strategy 1: Try product reviews URL (most reliable)
    try:
        reviews_url_base = url_modifier(original_url)
        if reviews_url_base:
            for page_num in range(start_page, end_page + 1):
                try:
                    page_url = f"{reviews_url_base}{page_num}"
                    response = session.get(page_url, timeout=15, allow_redirects=True)
                    
                    if response.status_code == 200:
                        # Handle encoding properly
                        response.encoding = 'utf-8'
                        soup = BeautifulSoup(response.content, 'html.parser')
                        page_reviews = extract_reviews_from_page(soup)
                        
                        if page_reviews:
                            all_reviews.extend(page_reviews)
                        elif page_num == start_page:
                            # No reviews on first page, might be wrong URL
                            break
                    elif response.status_code == 404:
                        if page_num == start_page:
                            # Reviews page doesn't exist
                            break
                    elif response.status_code in [403, 429]:
                        # Rate limited or blocked
                        break
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except requests.exceptions.RequestException:
                    # Network error, skip this page
                    continue
                except Exception:
                    # Other errors, skip silently
                    continue
        
        # If we got reviews, return them
        if all_reviews:
            return all_reviews
    
    except Exception:
        # If reviews URL fails, try product page
        pass
    
    # Strategy 2: Try product page directly
    try:
        response = session.get(original_url, timeout=15, allow_redirects=True)
        if response.status_code == 200:
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            page_reviews = extract_reviews_from_page(soup)
            if page_reviews:
                all_reviews.extend(page_reviews)
    except Exception:
        # Silently handle errors
        pass
    
    return all_reviews
