import streamlit as st
import re

def url_modifier(url):
    try:
        if not url or "flipkart.com" not in url.lower():
            return None
        
        # Remove query parameters and fragments, but keep the path
        url_parts = url.split('?')[0].split('#')[0]
        url_clean = url_parts.strip().rstrip('/')
        
        # Handle different Flipkart URL formats
        # Format 1: https://www.flipkart.com/product-name/p/productid
        # Format 2: https://www.flipkart.com/product-reviews/productid (already a reviews URL)
        # Format 3: https://flipkart.com/product-name/p/productid?pid=...
        
        # Check if it's already a reviews URL
        if "/product-reviews/" in url_clean:
            # Already a reviews URL, just add page parameter
            if "?" in url_clean:
                return url_clean + "&page="
            else:
                return url_clean + "?page="
        
        # Convert product URL to reviews URL
        if "/p/" in url_clean:
            # Extract product ID from /p/PRODUCTID pattern
            # Pattern: /p/PRODUCTID or /p/PRODUCTID? or /p/PRODUCTID/
            product_match = re.search(r'/p/([^/?]+)', url_clean)
            if product_match:
                product_id = product_match.group(1)
                # Remove any trailing slashes or parameters from product ID
                product_id = product_id.split('?')[0].split('/')[0]
                
                # Get the base URL
                if url_clean.startswith('http://'):
                    base = 'http://www.flipkart.com'
                elif url_clean.startswith('https://'):
                    base = 'https://www.flipkart.com'
                else:
                    base = 'https://www.flipkart.com'
                
                # Construct reviews URL
                url_modified = f"{base}/product-reviews/{product_id}?page="
                return url_modified
            else:
                # Fallback: simple replace
                url_modified = url_clean.replace("/p/", "/product-reviews/")
                if "?" in url_modified:
                    url_modified = url_modified.split('?')[0] + "?page="
                else:
                    url_modified = url_modified + "?page="
                return url_modified
        elif "/product/" in url_clean:
            url_modified = url_clean.replace("/product/", "/product-reviews/")
            if "?" in url_modified:
                url_modified = url_modified.split('?')[0] + "?page="
            else:
                url_modified = url_modified + "?page="
            return url_modified
        else:
            # Try to extract product ID from various patterns
            product_match = re.search(r'/([a-zA-Z0-9]+)/p/([^/?]+)', url_clean)
            if product_match:
                product_id = product_match.group(2).split('?')[0]
                base = 'https://www.flipkart.com'
                return f"{base}/product-reviews/{product_id}?page="
            
            # Last resort: try to find any product ID pattern
            product_match = re.search(r'pid=([^&]+)', url)
            if product_match:
                product_id = product_match.group(1)
                base = 'https://www.flipkart.com'
                return f"{base}/product-reviews/{product_id}?page="
            
            return None
        
    except Exception as e:
        return None
