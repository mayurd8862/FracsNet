import os
import re
import json
import aiohttp
import asyncio
import pandas as pd # Keep pandas for potential future use or internal processing if needed
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import hashlib
import nest_asyncio
from urllib.parse import urlparse
from textblob import TextBlob
import logging
import sys
from decimal import Decimal, ROUND_HALF_UP # For precise price formatting

# Import the new LLM
from langchain_groq import ChatGroq
# Import tabulate for nice table printing
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


nest_asyncio.apply() # Apply early for environments like Jupyter/async frameworks
load_dotenv()

# --- Logging Setup (Unchanged) ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger("MedCompareScript")
logger.setLevel(logging.INFO) # Set to DEBUG for more verbose output if needed
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("textblob").setLevel(logging.WARNING)

# --- Constants and Allowed Domains (Unchanged) ---
ALLOWED_DOMAINS = [
    "amazon.in", "flipkart.com", "1mg.com",
    "netmeds.com", "apollopharmacy.in", "pharmeasy.in",
    "medplusmart.com", "walmart.com"
]

# --- Helper Functions (Unchanged) ---
# correct_spelling, adjust_query, extract_domain, is_allowed_url,
# is_generic_result, is_valid_detail, refine_details,
# extract_product_details, extract_certifications, extract_price,
# extract_rating, clean_title, product_unavailable
# (All these functions remain exactly as you provided them above)
# --- Start of Helper Functions ---

def correct_spelling(text):
    """Attempts to correct spelling using TextBlob."""
    try:
        # Ensure TextBlob has downloaded corpora if running for the first time
        # May require: python -m textblob.download_corpora
        return str(TextBlob(text).correct())
    except Exception as e:
        logger.warning(f"Spelling correction failed: {e}")
        return text

def adjust_query(user_query):
    """Cleans and adjusts the user query for better search results."""
    corrected = correct_spelling(user_query)
    # Remove comparison/command keywords
    base = re.sub(r"\b(compare|vs|versus|suggest me \d+|show me|which is better|that are there|find|get me)\b", "", corrected, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    # Add context if not present
    if not any(domain in base.lower() for domain in ["amazon", "flipkart", "1mg", "netmeds", "apollo", "pharmeasy", "medplus", "walmart"]):
        # Add "buy online india" if the query doesn't already imply online purchase or location
        if not re.search(r"\b(buy|online|price|order|shop)\b", base, re.I):
             base = f"{base} buy online india"
    return base.strip()


def extract_domain(url):
    """Extracts the cleaned domain name from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '').replace('m.', '').split(':')[0]
        # Handle specific domain variations
        if domain == "apollo.pharmacy":
             return "apollopharmacy.in"
        if domain == "jiomart.com": # Example if added later
             return "jiomart.com"
        return domain
    except Exception:
        return ""

def is_allowed_url(url):
    """Checks if the URL's domain is in the allowed list."""
    domain = extract_domain(url)
    return any(allowed == domain for allowed in ALLOWED_DOMAINS)

def is_generic_result(title, url):
    """Checks if a SERP result points to a generic category/search page instead of a specific product."""
    generic_terms = [
        "bestsellers", "product price list", "shop online", "trusted online",
        "order medicine", "category", "all products", "health conditions",
        "store near you", "pharmacy online", "medicine list", "vitamins", "supplements",
        "multivitamins", "health care", "top deals", "offers", "discounts",
        "compare prices", "search results", "collections", "brand store" # Added more
    ]
    combined = (title + " " + url).lower()

    # Check if the title itself is too generic
    generic_titles = ["medicines", "health products", "online pharmacy", "vitamins", "supplements",
                      "vitamin c", "multivitamin", "health", "shop", "category", "search", "products"]
    if title.lower() in generic_titles:
        # Allow if URL seems specific despite generic title (common on some sites)
        if any(frag in url.lower() for frag in ["/dp/", "/p/", "/otc/", "/product/", "/non-prescriptions/", "/drug/"]):
            return False # Likely a product page despite generic SERP title
        logger.debug(f"Filtered generic Title: {title} | {url}")
        return True

    # Check for generic path segments suggesting category or search pages
    generic_paths = ["/category/", "/categories/", "/all/", "/list/", "/browse/", "/health-care/",
                     "/non-prescriptions/", "/fitness/", "/search", "/shop/", "/store/", "/products",
                     "/brand/", "/offers/"]
    if any(path in url.lower() for path in generic_paths):
        # Allow specific product patterns even within these paths
        if any(frag in url.lower() for frag in ["/dp/", "/p/", "/otc/", "/product/", "/drug/"]):
             return False
        logger.debug(f"Filtered generic URL by path segment: {url}")
        return True

    # Check for generic keywords in title/URL
    if any(term in combined for term in generic_terms):
        # Allow if URL seems specific despite generic term
        if any(frag in url.lower() for frag in ["/dp/", "/p/", "/otc/", "/product/", "/drug/"]):
             return False
        logger.debug(f"Filtered generic URL/Title by keyword: {title} | {url}")
        return True

    # Check for URLs that end exactly with domain (homepage)
    parsed_url = urlparse(url.lower())
    if parsed_url.path.strip('/') == '':
         logger.debug(f"Filtered homepage URL: {url}")
         return True

    # Check for query parameters indicating a search page
    if "search?" in url.lower() or "q=" in url.lower() or "query=" in url.lower() or "search_query=" in url.lower():
        if any(frag in url.lower() for frag in ["/dp/", "/p/", "/otc/", "/product/", "/drug/"]): # Allow product pages reached via search
             return False
        logger.debug(f"Filtered search results URL: {url}")
        return True

    return False


def is_valid_detail(sentence):
    """Checks if a scraped sentence is likely a meaningful product detail."""
    sentence = sentence.strip()
    # Slightly adjusted limits, main validation is by content
    if len(sentence) < 15 or len(sentence) > 400: # Increased max slightly
        return False

    stop_phrases = [
        # Basic navigation/policy
        "privacy policy", "terms of use", "return policy", "add to cart", "in stock",
        "buy now", "shop now", "go back", "view details", "learn more", "click here",
        "home delivery", "free delivery", "shipping information", "secure transaction",
        "customer reviews", "related products", "frequently bought together", "also viewed",
        "login", "register", "track order", "sign in", "your account",
        "contact us", "about us", "need help", "customer care",
        "powered by", "copyright", "all rights reserved",
        "compare prices", "check availability", "select options", "quantity",
        # More specific labels often before content, not content itself
        "specifications:", "highlights:", "description:", "features:", "key benefits:",
        "how to use:", "safety information:", "ingredients:", "directions for use:",
        "note:", "warning:", "caution:", "mrp:", "expiry date:", "manufacturer:", "marketer:",
        "subscribe", "newsletter", "download app", "visit store", "store near you",
        # Promotional / Empty phrases
        "tell us about", "lower price", "added to cart", "limited time offer", "best price guaranteed",
        "explore more", "view all", "read more", "view more",
        "frequently asked questions", "faq", "q&a",
        # Boilerplate / Legal
        "csrf", "important information", "legal disclaimer", "consult your physician", "ask doctor",
        "results may vary", "keep out of reach of children", "for external use only",
        # Website specific clutter
        "skip to main content", "search suggestions", "no results found", "page not found",
        "image is for representation purpose only", "mrp", "inclusive of all taxes",
        "sold by", "fulfilled by", "seller information", "country of origin",
        "netmeds first", "1mg verified", "apollo certified", # Avoid seller badges as details
        "you might be interested in", "sponsored", "advertisement", "from the manufacturer",
        "go to cart", "proceed to checkout", "apply coupon", "available offers",
        "share this product", "add to wishlist", # Social/Interaction
        "what is this?", "how does this work?", # Generic questions often used as headers
        # Fragments often resulting from bad parsing
        "javascript", "function(", "{", "}", "[", "]", "=>", "error code", "session timed out",
        "filter by", "sort by", "items per page", "out of 5 stars" # Rating fragments
    ]

    lower_sentence = sentence.lower()

    # Check for exact or near-exact matches with stop phrases
    if lower_sentence in [p.lower() for p in stop_phrases]:
        logger.debug(f"Filtered detail by exact stop phrase: {sentence[:50]}...")
        return False
    # Check if sentence *starts* with a stop phrase (common for list items or labels)
    if any(lower_sentence.startswith(phrase) for phrase in stop_phrases):
        logger.debug(f"Filtered detail by starting stop phrase: {sentence[:50]}...")
        return False
    # Check if sentence *contains* certain highly generic/spammy phrases
    spammy_contained = ["click here", "buy now", "add to cart", "limited time offer", "best price", "shop now", "free delivery", "customer reviews", "related products", "download our app", "subscribe to"]
    if any(phrase in lower_sentence for phrase in spammy_contained):
        logger.debug(f"Filtered detail by contained stop phrase: {sentence[:50]}...")
        return False

    # Check for list markers or excessive symbols often found in non-prose content
    if re.match(r"^\W*(\*|\-|•|>|\d+\.|\u2022)\s+", sentence): # Starts with list markers
        logger.debug(f"Filtered detail by list marker: {sentence[:50]}...")
        return False
    if sentence.count('{') > 1 or sentence.count('}') > 1 or sentence.count('[') > 2 or sentence.count(']') > 2: # Allow a few brackets
        logger.debug(f"Filtered detail by excessive brackets: {sentence[:50]}...")
        return False
    # Avoid URL-like strings, code snippets, or excessive numbers/slashes/colons
    if 'http:' in lower_sentence or 'https:' in lower_sentence or '.js' in lower_sentence:
        logger.debug(f"Filtered detail by URL/script content: {sentence[:50]}...")
        return False
    if sentence.count('/') > 5 or sentence.count(':') > 4 or len(re.findall(r'\d', sentence)) > len(sentence) * 0.7: # 70% numbers threshold
        logger.debug(f"Filtered detail by excessive symbols/numbers: {sentence[:50]}...")
        return False
    # Avoid sentences that are mostly uppercase (often headers or disclaimers)
    if sum(1 for c in sentence if c.isupper()) > len(sentence) * 0.7 and len(sentence) > 20: # Higher threshold, min length
        logger.debug(f"Filtered detail by uppercase dominance: {sentence[:50]}...")
        return False

    # Check for overly promotional or generic website language
    promo_patterns = [r"(?:best|top|great|amazing|fantastic)\s+(?:deals?|offers?|price|quality|value)",
                      r"shop\s+our\s+(?:wide\s+)?selection",
                      r"lowest\s+prices?\s+guaranteed",
                      r"100%\s+(?:authentic|genuine|original)", # Keep genuine product description though
                      r"satisfaction\s+guaranteed"]
    if any(re.search(pattern, lower_sentence) for pattern in promo_patterns):
        # Allow if it *also* contains specific health keywords, maybe it's describing benefits
        health_keywords = ["vitamin", "mineral", "health", "immune", "skin", "hair", "energy", "relief"]
        if not any(kw in lower_sentence for kw in health_keywords):
            logger.debug(f"Filtered detail by promo language: {sentence[:50]}...")
            return False

    # Filter out short fragments that often come from image captions or isolated labels
    # Removed the verb check - too restrictive
    if len(sentence.split()) < 5: # Increase minimum words slightly
         logger.debug(f"Filtered short detail fragment: {sentence[:50]}...")
         return False

    # Filter out sentences that look like address details
    if re.search(r'\b(street|road|nagar|colony|pincode|district|state|city)\b', lower_sentence, re.I) and len(sentence.split()) < 15:
         logger.debug(f"Filtered detail potentially address: {sentence[:50]}...")
         return False

    # If it passes all checks, it's considered valid
    logger.debug(f"VALID detail: {sentence[:50]}...")
    return True

def refine_details(details):
    """Cleans, filters, deduplicates, and selects the best product details."""
    refined = []
    for d in details:
        # Clean up common residues and normalize whitespace
        d = d.replace("View more", "").replace("Read more", "").strip()
        d = re.sub(r'\s+', ' ', d).strip()
        # Additional cleanup for specific patterns
        d = re.sub(r'^\d+\.\s*', '', d) # Remove leading "1. ", "2. "
        d = re.sub(r'^-\s*', '', d) # Remove leading "- "
        d = re.sub(r'^\*\s*', '', d) # Remove leading "* "
        if d and is_valid_detail(d): # Apply the validation function
             refined.append(d)

    seen_simplified = set()
    deduplicated = []
    for d in refined:
        # Use a simplified version for checking duplicates (lowercase, no punctuation/spaces)
        simplified_d = ''.join(filter(str.isalnum, d.lower()))
        if not simplified_d: continue # Skip if simplification results in empty string

        is_near_duplicate = False
        # Check against already added simplified versions
        for s_seen in seen_simplified:
            # Check if one is a substring of the other (very common for near duplicates)
            # Or if they are highly similar (using a basic length check as proxy)
            if (simplified_d in s_seen or s_seen in simplified_d) and \
               abs(len(simplified_d) - len(s_seen)) < max(len(simplified_d), len(s_seen)) * 0.2: # Allow only 20% length diff for substrings
                 is_near_duplicate = True
                 logger.debug(f"Filtered near-duplicate detail: '{d[:50]}...' similar to seen.")
                 break
        if not is_near_duplicate:
            seen_simplified.add(simplified_d)
            deduplicated.append(d)

    # Sort by relevance (heuristic: longer details might be more informative, place health-related higher)
    keywords = ["vitamin", "probiotic", "nutrition", "tablet", "capsule", "supplement", "digest", "health", "immune", "immunity", "strength", "energy", "skin", "hair", "bone", "joint", "heart", "brain", "gut", "relief", "support", "mg", "ml", "ingredient", "extract", "mineral", "herbal", "ayurvedic", "benefit", "helps", "provides", "contains", "improves"]
    def sort_key_detail(text):
        score = len(text)
        if any(kw in text.lower() for kw in keywords):
            score += 100 # Boost score for health-related keywords
        return score

    deduplicated.sort(key=sort_key_detail, reverse=True)

    # Select top details, maybe limit to 3-5?
    final_details = deduplicated[:4] # Limit to top 4 after sorting and deduplication

    logger.info(f"Refined details count: {len(final_details)} (from initial {len(details)})")
    return final_details


def extract_product_details(soup, domain):
    """Extracts product description and feature details from the page."""
    details = []
    processed_texts_simplified = set() # Use simplified text for tracking

    # Helper to add detail if valid and not already processed
    def add_detail(text):
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if not cleaned_text: return

        simplified_text = ''.join(filter(str.isalnum, cleaned_text.lower()))
        if simplified_text and simplified_text not in processed_texts_simplified:
            # Validation will happen later in refine_details, just collect candidates here
            details.append(cleaned_text)
            processed_texts_simplified.add(simplified_text)

    # 1. Try ld+json description extraction (High Priority)
    try:
        found_ld_json_desc = False
        ld_json_scripts = soup.find_all("script", type="application/ld+json")
        logger.debug(f"Found {len(ld_json_scripts)} ld+json scripts.")
        for script in ld_json_scripts:
            try:
                data = json.loads(script.string)
                queue = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
                while queue:
                    item = queue.pop(0)
                    if isinstance(item, dict):
                        item_type = item.get("@type")
                        # Look for description in relevant types
                        if item_type in ["Product", "Drug", "ProductGroup", "SomeProducts"]: # Added more types
                            desc = item.get("description") or item.get("detailedDescription")
                            if desc and isinstance(desc, str):
                                logger.debug(f"Found description in LD+JSON ({item_type}): {desc[:100]}...")
                                # Split into sentences carefully using regex, handle HTML within desc
                                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\d])|(?<=[.!?])\s*<br\s*/?>\s*|[\n\r]+', desc)
                                for sentence in sentences:
                                    sentence_text = BeautifulSoup(sentence, 'html.parser').get_text(" ", strip=True)
                                    add_detail(sentence_text)
                                found_ld_json_desc = True

                        # Check nested structures like graph, mainEntity, etc.
                        for key in ["@graph", "mainEntity", "subjectOf", "detailedDescription", "hasPart"]: # Added hasPart
                           if key in item:
                               potential_data = item[key]
                               if isinstance(potential_data, list): queue.extend(potential_data)
                               elif isinstance(potential_data, dict): queue.append(potential_data)
                    elif isinstance(item, list): queue.extend(item)
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                # logger.warning(f"LD+JSON parsing error: {e}") # Can be noisy
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing script tag content: {e}")
    except Exception as e:
        logger.error(f"General error processing script tags: {e}")

    # 2. Domain-specific extraction - Run even if ld+json found, to add more structured data like features
    specific_details = []
    try:
        if "amazon.in" in domain:
            # Feature bullets are often good details
            bullets_div = soup.find("div", id="feature-bullets")
            if bullets_div:
                for li in bullets_div.find_all("li"):
                    text = li.get_text(" ", strip=True)
                    # Filter out generic bullet points often added by Amazon
                    if text and not text.lower().startswith(("see more product details", "country of origin", "customer reviews")):
                        specific_details.append(text)
                        logger.debug(f"Amazon feature bullet added: {text[:50]}...")
            # Product description section (if no bullets, or to supplement)
            desc_div = soup.find("div", id="productDescription")
            if desc_div:
                paragraphs = desc_div.find_all("p", limit=5) # Limit paragraphs
                for p in paragraphs: specific_details.append(p.get_text(" ", strip=True))

        elif "flipkart.com" in domain:
            # Highlights / Key Features (Classes change often!) - Combine known patterns
            highlights_div = soup.find("div", class_=lambda x: x and ('_2418kt' in x or 'rukqdk' in x or '_1FHujX' in x or '_1YokD2' in x)) # Added _1YokD2
            if highlights_div:
                for li in highlights_div.find_all(["li", "div"], class_=lambda x: x and ('_21Ahn-' in x or '_21lJbe' in x), limit=5): # Limit items
                    item_text = li.get_text(" ", strip=True)
                    if item_text: specific_details.append(item_text)
                    logger.debug(f"Flipkart highlight/feature added: {item_text[:50]}...")
            # Detailed description section
            desc_div = soup.find("div", class_=lambda x: x and ('_1AN87F' in x or '_1mXcCf' in x))
            if desc_div:
                desc_text = desc_div.get_text(" ", strip=True)
                # Split potentially long text into sentences if needed
                if len(desc_text) > 200:
                    sentences = re.split(r'(?<=[.!?])\s+', desc_text)
                    specific_details.extend(s for s in sentences if len(s) > 10)
                elif len(desc_text) > 10:
                    specific_details.append(desc_text)

        # --- Pharmacy Sites ---
        elif any(pharma in domain for pharma in ["1mg.com", "netmeds.com", "apollopharmacy.in", "pharmeasy.in"]):
            processed_section_content_simplified = set()
            section_headings = ["Product Information", "Key Benefits", "Product Details", "Benefits", "Uses", "Features", "Description", "Directions for Use", "Safety Information", "Key Ingredients", "Uses of", "Benefits of", "Side effects of", "How to use"] # Added more variations
            container_selectors = [
                "div[class*='description']", "div[class*='details']", "div[class*='info']",
                "div[class*='content']", "div[class*='ProductDescription']", "div[class*='ProductPage__Info']",
                "div[class*='PdpWeb_productInfo']", "section[class*='Product']", "div.drug-info",
                "div#accor"] # Targeting accordion wrappers too

            section_containers = []
            for sel in container_selectors:
                section_containers.extend(soup.select(sel))
            if not section_containers: section_containers = [soup.body] # Fallback

            for container in section_containers:
                if not container: continue
                # Find headings within the container first
                headings = container.find_all(['h2', 'h3', 'h4', 'strong', 'div', 'span'],
                                              string=re.compile(r'|'.join(section_headings), re.I), limit=10)
                # If no string match, try classes
                if not headings:
                    headings = container.find_all(['h2', 'h3', 'h4', 'strong', 'div', 'span'],
                                                  class_=re.compile(r'heading|title|header|lbl', re.I), limit=10)

                if headings:
                    for heading in headings:
                        # Find associated content: often next sibling(s) or content within sibling div
                        content_node = None
                        current_node = heading
                        found_content_nodes = []
                        # Look for next few siblings until another heading or irrelevant tag
                        for _ in range(4): # Check next 4 siblings max
                            sibling = current_node.find_next_sibling()
                            if not sibling or sibling.name in ['h2','h3','h4','script','style','button','form'] or sibling.find(['h2','h3','h4']): # Stop at next heading or non-content
                                break
                            # Collect relevant content tags
                            if sibling.name in ['p', 'div', 'ul', 'span', 'table'] and sibling.get_text(strip=True):
                                found_content_nodes.append(sibling)
                            current_node = sibling

                        if found_content_nodes:
                             for node in found_content_nodes:
                                 if node.name == 'ul':
                                     for li in node.find_all('li', recursive=False, limit=5): # Limit list items
                                         item_text = li.get_text(" ", strip=True)
                                         if item_text:
                                             simplified = ''.join(filter(str.isalnum, item_text.lower()))
                                             if simplified not in processed_section_content_simplified:
                                                 specific_details.append(item_text)
                                                 processed_section_content_simplified.add(simplified)
                                                 logger.debug(f"Pharmacy section list item added: {item_text[:50]}...")
                                 else:
                                     content_text = node.get_text(" ", strip=True)
                                     simplified = ''.join(filter(str.isalnum, content_text.lower()))
                                     if simplified not in processed_section_content_simplified:
                                         specific_details.append(content_text)
                                         processed_section_content_simplified.add(simplified)
                                         logger.debug(f"Pharmacy section content added: {content_text[:50]}...")
                else: # If no headings found in container, try extracting paragraphs directly
                     paragraphs = container.find_all("p", limit=3)
                     for p in paragraphs:
                         para_text = p.get_text(" ", strip=True)
                         simplified = ''.join(filter(str.isalnum, para_text.lower()))
                         if simplified not in processed_section_content_simplified:
                             specific_details.append(para_text)
                             processed_section_content_simplified.add(simplified)
                             logger.debug(f"Pharmacy container paragraph added: {para_text[:50]}...")


            # Fallback for specific description classes if sections yielded little
            if len(specific_details) < 2 and not found_ld_json_desc: # Only fallback if few details AND no LD+JSON found
                 fallback_selectors = [
                     "div[class*='ProductDescription__description']", # 1mg
                     "div[class*='Description']", # Generic
                     "div#product_content .inner-content p", # Netmeds paragraphs
                     "div[class*='PdpWeb_productInfo__'] p", "div[class*='ProductDetailsGeneric_desc'] p", # Apollo paragraphs
                     "div[class*='ProductDescriptionContainer_description'] p", "div[class*='ProductPage__InfoContainer'] p" # Pharmeasy paragraphs
                 ]
                 for sel in fallback_selectors:
                     desc_elements = soup.select(sel, limit=3)
                     for element in desc_elements:
                         fallback_text = element.get_text(" ", strip=True)
                         simplified = ''.join(filter(str.isalnum, fallback_text.lower()))
                         if simplified not in processed_section_content_simplified:
                             specific_details.append(fallback_text)
                             processed_section_content_simplified.add(simplified)
                             logger.debug(f"Pharmacy fallback detail added: {fallback_text[:50]}...")
                     if len(specific_details) >= 2: break # Stop fallback if we got enough


        # --- Medplusmart ---
        elif "medplusmart.com" in domain:
            # Often uses specific divs or tables for details
             info_section = soup.select_one('div.product-more-info') or soup.select_one("div.prodDetails") # Example classes
             if info_section:
                 # Try paragraphs first
                 paragraphs = info_section.find_all('p', limit=3)
                 for p in paragraphs: specific_details.append(p.get_text(" ", strip=True))
                 # Try list items
                 list_items = info_section.find_all('li', limit=5)
                 for li in list_items: specific_details.append(li.get_text(" ", strip=True))
                 # Try table data if present
                 for td in info_section.find_all('td', limit=5):
                     td_text = td.get_text(" ", strip=True)
                     # Avoid adding labels like "Uses", "Benefits" as details themselves
                     if td_text and len(td_text.split()) > 2: # Basic check to avoid short labels
                         specific_details.append(td_text)

        # Add collected specific details to the main list if they haven't been processed already
        for detail_text in specific_details:
            add_detail(detail_text) # add_detail handles deduplication via processed_texts_simplified

    except Exception as e:
         logger.warning(f"Error during domain-specific detail extraction for {domain}: {e}", exc_info=True) # Show traceback for warnings


    # 3. Generic Paragraph Extraction (Lowest Priority) - Only if very few details found so far
    if len(details) < 3 and not found_ld_json_desc: # More strict condition: only if few details and no structured data
        try:
            # Look within main content areas first
            main_content = soup.find("main") or soup.find("article") or soup.find("div", id="dp-container") or soup.find("div", id="content") or soup.find("div", role="main") or soup.body
            if main_content:
                # Find paragraphs that are not obviously inside nav, header, footer, or script/style tags
                paragraphs = main_content.find_all("p", limit=10) # Limit search space
                for p in paragraphs:
                    # Check if paragraph is inside a non-relevant parent
                    is_irrelevant = False
                    for parent in p.parents:
                         # Added more irrelevant tags like form, button, figure, aside
                        if parent.name in ['nav', 'header', 'footer', 'aside', 'script', 'style', 'form', 'button', 'figure']:
                            is_irrelevant = True
                            break
                         # Stop checking after a few levels up
                        if parent.name == 'body': break
                    if is_irrelevant: continue

                    text = p.get_text(" ", strip=True)
                    if len(text.split()) > 5: # Ensure paragraph has some substance
                         add_detail(text) # Add detail (validation happens in refine_details)
        except Exception as e:
             logger.warning(f"Error during generic paragraph extraction: {e}")

    # 4. Refine and Filter ALL collected details
    refined_and_filtered = refine_details(details)

    return refined_and_filtered


# --- Certification Extraction (Unchanged) ---
def extract_certifications(soup):
    """Extracts potential certification mentions from the page."""
    certs = set()
    keywords = ["certified", "iso", "fssai", "approved", "authentic", "verified", "legitscript", "gmp", "non-gmo", "organic", "ayush", "fda", "lab tested", "halal", "kosher", "usda"] # Added more keywords
    processed_cert_texts_simplified = set()

    def add_cert(text):
        cleaned = re.sub(r'\s+', ' ', text).strip()
        # Filter out overly long texts or generic sentences that happen to contain a keyword
        # Also filter very short mentions that might be part of other text
        if cleaned and 8 < len(cleaned) < 100 and cleaned.count('.') <= 1 and cleaned.count(',') <= 1:
            simplified = ''.join(filter(str.isalnum, cleaned.lower()))
            # Additional filter: avoid adding if it looks like a normal sentence fragment
            common_words = ["is", "are", "the", "a", "an", "and", "or", "but", "product", "medicine"]
            if simplified and simplified not in processed_cert_texts_simplified and not any(word in cleaned.lower().split() for word in common_words):
                 certs.add(cleaned)
                 processed_cert_texts_simplified.add(simplified)
                 logger.debug(f"Potential certification added: {cleaned}")

    try:
        # Search for text containing keywords within specific tags first
        # Look in spans, divs, paragraphs, list items, strong tags near keywords
        potential_cert_tags = soup.find_all(['p', 'div', 'span', 'li', 'em', 'strong'], limit=150) # Limit search scope
        for tag in potential_cert_tags:
            text = tag.get_text(" ", strip=True)
            lower_text = text.lower()
            if any(kw in lower_text for kw in keywords):
                 # Check if the text is concise and likely a certification mention
                 if len(text.split()) < 7: # Only add short phrases containing keyword
                     add_cert(text)

        # Look for images with alt text suggesting certification
        for img in soup.find_all("img", alt=True, limit=30):
             alt_text = img.get('alt', '')
             title_text = img.get('title', '')
             combined_text = f"{alt_text} {title_text}".strip()
             if any(kw in combined_text.lower() for kw in keywords):
                 if len(combined_text.split()) < 7: # Only add short alt/title texts
                     add_cert(combined_text)

        # Check specific attributes known to sometimes contain certifications
        # Added classes like 'badge', 'label', 'tag'
        attr_selectors = ['[data-testid*="badge"]', '[data-testid*="certifi"]', '[class*="badge"]', '[class*="label"]', '[class*="tag"]', '[class*="certified"]', '[class*="verified"]']
        for sel in attr_selectors:
             elements = soup.select(sel, limit=10)
             for tag in elements:
                 tag_text = tag.get_text(" ", strip=True)
                 if any(kw in tag_text.lower() for kw in keywords):
                     if len(tag_text.split()) < 7:
                         add_cert(tag_text)

    except Exception as e:
         logger.warning(f"Error extracting certifications: {e}")

    # Refine: Remove duplicates / overly similar certs
    refined_certs_list = list(certs)
    refined_certs_list.sort(key=len) # Process shorter ones first

    final_certs = []
    seen_simplified_final = set()
    for cert in refined_certs_list:
        simplified = ''.join(filter(str.isalnum, cert.lower()))
        is_duplicate = False
        for seen in seen_simplified_final:
            if simplified in seen or seen in simplified:
                is_duplicate = True
                break
        if not is_duplicate:
            # Filter out generic purchase/buyer related "certifications" more strictly
            if not any(phrase in cert.lower() for phrase in ["verified purchase", "certified buyer", "authentic product", "genuine product", "secure transaction", "trusted seller", "assured", "ratings based", "reviews", "popularity"]):
                 final_certs.append(cert)
                 seen_simplified_final.add(simplified)

    logger.info(f"Found {len(final_certs)} potential certifications.")
    return final_certs[:3] # Return max 3 distinct certifications


# --- Price Extraction (Unchanged) ---
def extract_price(soup):
    """Extracts the product price, prioritizing specific elements and LD+JSON."""
    price_text = "Price not available"
    found_price = None

    # Helper function to format price string using Decimal
    def format_price(value_str):
        try:
            # Remove currency symbols (₹, Rs, INR), commas, spaces, and any trailing noise like '/-'
            cleaned_val = re.sub(r"[^0-9.]", "", value_str.split('/')[0])
            if not cleaned_val: return None
            # Use Decimal for accurate financial calculations/formatting
            price_decimal = Decimal(cleaned_val)
            # Format to two decimal places, using Indian Rupee symbol
            formatted = f"₹{price_decimal.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"
            # Basic sanity check (e.g., price shouldn't be zero or excessively high)
            if 0 < price_decimal < 100000: # Adjust upper limit if needed
                return formatted
            else:
                 logger.debug(f"Filtered out potential price outside valid range: {value_str} -> {price_decimal}")
                 return None
        except Exception as e: # Handle potential conversion errors
            logger.debug(f"Price formatting failed for '{value_str}': {e}")
            return None

    # 1. High-priority: Meta tags (including ld+json)
    try:
        # Check ld+json first for Offer or Product price
        ld_json_scripts = soup.find_all("script", type="application/ld+json")
        logger.debug(f"Checking {len(ld_json_scripts)} ld+json scripts for price...")
        for script in ld_json_scripts:
             try:
                 data = json.loads(script.string)
                 queue = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
                 while queue:
                     item = queue.pop(0)
                     if isinstance(item, dict):
                         item_type = item.get("@type")
                         offers_data = None
                         potential_price_val = None
                         currency = None

                         # Prioritize offers within Product/Drug first
                         if item_type in ["Product", "Drug"]:
                             offers_data = item.get("offers")
                         # Then check Offer/AggregateOffer directly
                         elif item_type in ["Offer", "AggregateOffer"]:
                             offers_data = item # Treat the item itself as the offer

                         # Process Offers if found
                         if offers_data:
                             offer_queue = [offers_data] if isinstance(offers_data, dict) else offers_data if isinstance(offers_data, list) else []
                             while offer_queue:
                                 offer_item = offer_queue.pop(0)
                                 if isinstance(offer_item, dict):
                                     offer_type = offer_item.get("@type")
                                     if offer_type in ["Offer", "AggregateOffer"]:
                                         offer_price = offer_item.get("price") or offer_item.get("lowPrice")
                                         offer_currency = offer_item.get("priceCurrency")
                                         availability = offer_item.get("availability", "").lower()
                                         # Check if offer is valid (in stock, correct currency)
                                         if offer_price and offer_currency in ["INR", "Rs", "₹"] and 'instock' in availability or 'onlineonly' in availability or not availability:
                                             formatted = format_price(str(offer_price))
                                             if formatted:
                                                 logger.info(f"Price found via LD+JSON ({offer_type}): {formatted}")
                                                 return formatted # Prioritize valid offer price

                                 elif isinstance(offer_item, list): # Handle lists of offers
                                     offer_queue.extend(offer_item)

                         # If no offer price found, check Product/Drug level price (less common but possible)
                         if item_type in ["Product", "Drug"]:
                             potential_price_val = item.get("price") or item.get("value") # Check 'value' too
                             currency = item.get("priceCurrency") or item.get("currency")
                             if potential_price_val and currency in ["INR", "Rs", "₹"]:
                                 formatted = format_price(str(potential_price_val))
                                 if formatted and not found_price: # Only take if no offer price was better
                                     found_price = formatted
                                     logger.info(f"Price found via LD+JSON ({item_type} base): {formatted}")


                         # Check nested structures
                         for key in ["@graph", "mainEntity", "subjectOf", "hasPart"]:
                             if key in item:
                                 potential_data = item[key]
                                 if isinstance(potential_data, list): queue.extend(potential_data)
                                 elif isinstance(potential_data, dict): queue.append(potential_data)

                     elif isinstance(item, list): queue.extend(item)
             except Exception as e:
                 # logger.warning(f"LD+JSON price parsing error: {e}")
                 continue

        # Check standard meta tags if LD+JSON didn't yield a price
        if not found_price:
             meta_selectors = [ {"itemprop": "price"}, {"property": "product:price:amount"}, {"property": "og:price:amount"} ]
             for sel in meta_selectors:
                 meta = soup.find("meta", attrs=sel)
                 if meta and meta.get("content"):
                     formatted = format_price(meta["content"])
                     if formatted:
                         logger.info(f"Price found via meta tag {sel}: {formatted}")
                         return formatted # Found via standard meta

    except Exception as e:
        logger.warning(f"Error finding price in meta/LD+JSON: {e}")

    # Return price if found in LD+JSON earlier (e.g., base product price)
    if found_price: return found_price

    # 2. Common price elements (Selectors need constant updates!)
    try:
        # Combined list of selectors, prioritized roughly
        selectors = [
            # Specific IDs/Classes known for price (High Priority - check inspect element often!)
            "#priceblock_ourprice", "#priceblock_dealprice", # Amazon old
            ".priceToPay span.a-price-whole", ".a-price .a-offscreen", # Amazon current (offscreen is often the one)
            "span.priceToPay", "span.savingPriceOverride", # Amazon other possibilities
            "div._30jeq3._16Jk6d", "div._30jeq3", "div.Nx9bqj", # Flipkart (check current classes)
            ".Price__price___32PR5", ".style__price-tag___KzOkY", ".PriceDetails__final-price___Q7259", # 1mg older examples
            ".style__discount-price___", ".ProductPrice_finalPrice__", ".ProductPrice_price__", "[class*='product-price']", # Pharmeasy, 1mg newer? (more generic)
            ".PriceInfo_finalPrice__", ".style__value___", ".PdpWeb_price__", ".PdpWeb_sellingPrice__", # Apollo examples
            ".final-price", ".drug-price .price", ".drug-price", ".PriceDetails__ItemPrice", # Netmeds examples
            ".price-box .price", ".product-info-price .price", # Medplus examples?
            # General Attributes / Classes (Lower Priority)
            "[itemprop='price']", "[itemprop='lowPrice']", "[data-price]",
            "[class*='Price-value']", "[class*='price-final']", "[class*='selling-price']",
            "[class*='special-price']", "[class*='offer-price']",
            # Try spans/divs containing currency symbol directly (check text content)
            "span:contains('₹')", "div:contains('₹')", "span:contains('Rs')", "div:contains('Rs')" # BS4 :contains might be slow, use carefully
        ]

        for sel in selectors:
             elements = soup.select(sel, limit=5) # Limit elements per selector
             for element in elements:
                 # Extract text, prefer specific elements within if possible
                 text = element.get_text(strip=True)
                 content_attr = element.get('content', '') # Check content attribute too

                 potential_price_val = None
                 # Prioritize Amazon's hidden price span
                 if "a-offscreen" in element.get("class", []) and text.startswith("₹"):
                     potential_price_val = text
                 # Prioritize content attribute if it looks like a price
                 elif content_attr and re.search(r"^\d+(\.\d+)?$", content_attr):
                     potential_price_val = content_attr
                 # Then use element text if it contains currency
                 elif text and ('₹' in text or 'Rs' in text.upper() or 'INR' in text.upper()):
                      potential_price_val = text
                 # Fallback: Check if element is a price attribute holder and has content
                 elif element.has_attr('itemprop') and element['itemprop'] in ['price', 'lowPrice'] and content_attr:
                      potential_price_val = content_attr

                 if potential_price_val:
                     formatted = format_price(potential_price_val)
                     if formatted:
                         logger.info(f"Price found via element selector '{sel}': {formatted}")
                         return formatted # Return first valid price found

    except Exception as e:
        logger.warning(f"Error extracting price from common elements: {e}")

    return price_text # Return default if not found


# --- Rating Extraction (Unchanged) ---
def extract_rating(soup, domain):
    """Extracts product rating, trying to determine the scale (e.g., /5 or /10)."""
    rating_text = "Rating not available"
    rating_pattern = r"(\d(?:\.\d+)?)" # Matches rating number like 4 or 4.3

    # Helper function to validate and format rating
    def format_rating(val_str, scale_str="5"): # Default scale is 5
        try:
            val = float(val_str)
            # Try to extract scale number, default to 5 if invalid
            scale_match = re.search(r"(\d+)", str(scale_str))
            scale = int(scale_match.group(1)) if scale_match else 5
            if scale not in [5, 10]: scale = 5 # Ensure scale is 5 or 10

            # Validate value against scale
            if 0 <= val <= scale:
                # Format consistently (e.g., 4.0/5, 4.3/5)
                return f"{val:.1f}/{scale}" # Ensure one decimal place
            else:
                 logger.debug(f"Rating value {val} out of range for scale {scale}.")
                 return None
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Rating formatting/validation failed for '{val_str}' / '{scale_str}': {e}")
            return None

    # 1. High-priority: Meta tags (including ld+json)
    try:
        # Check ld+json first for AggregateRating
        ld_json_scripts = soup.find_all("script", type="application/ld+json")
        logger.debug(f"Checking {len(ld_json_scripts)} ld+json scripts for rating...")
        for script in ld_json_scripts:
             try:
                 data = json.loads(script.string)
                 queue = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
                 while queue:
                     item = queue.pop(0)
                     if isinstance(item, dict):
                         rating_data = None
                         item_type = item.get("@type")

                         # Check for AggregateRating directly or nested within Product/Drug
                         if item_type == "AggregateRating":
                             rating_data = item
                         elif item_type in ["Product", "Drug"] and isinstance(item.get("aggregateRating"), dict):
                             rating_data = item["aggregateRating"]

                         if rating_data:
                             rating_val = rating_data.get("ratingValue")
                             best_rating = rating_data.get("bestRating") or rating_data.get("ratingScale") # Check both
                             review_count = rating_data.get("reviewCount") or rating_data.get("ratingCount")

                             # Require ratingValue and ideally some reviews to be valid
                             if rating_val and (review_count is None or int(review_count) > 0): # Allow if reviewCount is missing, but > 0 if present
                                 scale = "5" # Default scale
                                 if best_rating:
                                     scale_match = re.search(r"(\d+)", str(best_rating))
                                     if scale_match and int(scale_match.group(1)) in [5, 10]:
                                         scale = scale_match.group(1)

                                 formatted = format_rating(str(rating_val), scale)
                                 if formatted:
                                     logger.info(f"Rating found via LD+JSON (AggregateRating): {formatted}")
                                     return formatted # Found via ld+json AggregateRating

                         # Check nested structures
                         for key in ["@graph", "mainEntity", "subjectOf", "review", "hasPart"]: # Added review
                             if key in item:
                                 potential_data = item[key]
                                 if isinstance(potential_data, list): queue.extend(potential_data)
                                 elif isinstance(potential_data, dict): queue.append(potential_data)

                     elif isinstance(item, list): queue.extend(item)
             except Exception as e:
                 # logger.warning(f"LD+JSON rating parsing error: {e}")
                 continue

        # Check standard meta tags if LD+JSON failed
        meta_rating = soup.find("meta", {"itemprop": "ratingValue"})
        if meta_rating and meta_rating.get("content"):
            val_str = meta_rating["content"]
            meta_max = soup.find("meta", {"itemprop": "bestRating"}) or soup.find("meta", {"itemprop": "ratingScale"})
            scale = "5" # Default
            if meta_max and meta_max.get("content"):
                 scale_match = re.search(r"(\d+)", meta_max.get('content'))
                 if scale_match and int(scale_match.group(1)) in [5, 10]:
                     scale = scale_match.group(1)

            # Check review count meta tag for validation
            meta_count = soup.find("meta", {"itemprop": "reviewCount"}) or soup.find("meta", {"itemprop": "ratingCount"})
            if meta_count and meta_count.get("content"):
                try:
                     if int(meta_count.get("content")) <= 0: # Skip if review count is 0
                         logger.debug("Skipping meta rating due to 0 reviews.")
                         pass # Continue searching
                     else:
                         formatted = format_rating(val_str, scale)
                         if formatted:
                             logger.info(f"Rating found via meta tags: {formatted}")
                             return formatted # Found via standard meta
                except (ValueError, TypeError): pass # Ignore count errors
            else: # If no count tag, format anyway
                 formatted = format_rating(val_str, scale)
                 if formatted:
                     logger.info(f"Rating found via meta tags (no count): {formatted}")
                     return formatted


    except Exception as e:
         logger.warning(f"Error finding rating in meta/LD+JSON: {e}")

    # 2. Domain-specific and common elements
    try:
        # Selectors ordered by likelihood of containing clear rating info
        selectors = [
            # Explicit "X out of Y" patterns
            "#acrPopover .a-icon-alt", # Amazon: "4.2 out of 5 stars"
            "span[data-hook='rating-out-of-text']", # Amazon alternative
            "div[class*='rating'] span:contains('out of')", # Generic pattern
            # Specific rating elements (Check inspect element)
            "div._3LWZlK", # Flipkart: "4.3 ★" - Assume /5 if only number+star
            ".Rating__rating-number___", ".CardRating_ratings__", ".styles__prodRating___", # 1mg examples
            ".RatingWrapper__ratings-container___", ".RatingStrip__ratings___",
            ".ProductRating_ratingNumber__", # Pharmeasy example
            ".Rating_container__", ".ReviewAndRating_ratingsValue__", # Apollo example
            ".drug-ratings span", # Netmeds example
            "span[itemprop='ratingValue']", # Sometimes value is directly in span
            # General classes containing 'rating', 'star' (Lower priority, check context)
            "[class*='ratingValue']", "[class*='RatingValue']",
            "[class*='rating-']", "[class*='Rating-']", # Hyphenated
            "[class*='star']", "[class*='Star']"
        ]

        for sel in selectors:
            elements = soup.select(sel, limit=5)
            for element in elements:
                text = element.get_text(" ", strip=True)
                aria_label = element.get('aria-label', '')
                title_attr = element.get('title', '')

                # Combine potential sources of rating text
                full_text = f"{text} {aria_label} {title_attr}".strip()
                if not full_text: continue

                logger.debug(f"Checking rating text: '{full_text}' from selector '{sel}'")

                # Try matching "X out of Y" pattern first (most reliable)
                match_out_of = re.search(r"(\d\.?\d+)\s*(?:out\s+of|/)\s*(\d+)", full_text, re.IGNORECASE)
                if match_out_of:
                    val_str, scale_str = match_out_of.groups()
                    formatted = format_rating(val_str, scale_str)
                    if formatted:
                        logger.info(f"Rating found via 'out of' pattern in element: {formatted}")
                        return formatted

                # If no explicit scale, match the number and infer scale=5 if context is right
                match_num = re.search(rating_pattern, full_text)
                if match_num:
                    val_str = match_num.group(1)
                    # Infer scale 5, but ONLY if context strongly suggests it's a rating
                    # Requires rating/star keyword OR symbol, plausible value, and NOT matching the 'out of' pattern above
                    if (('rating' in full_text.lower() or 'star' in full_text.lower() or '★' in full_text)
                        and float(val_str) <= 5): # Check if value is within 0-5 range
                         # Also check if it's just the number without "out of" nearby
                         if "out of" not in full_text.lower() and "/" not in full_text:
                             formatted = format_rating(val_str, "5") # Infer scale 5
                             if formatted:
                                 logger.info(f"Rating found via number pattern (inferred /5): {formatted}")
                                 return formatted

    except Exception as e:
        logger.warning(f"Error extracting rating from common elements: {e}")

    return rating_text # Return default if not found


# --- Title Cleaning (Unchanged) ---
def clean_title(title, soup):
    """Cleans the product title from SERP and page elements."""
    # 1. Initial cleaning from SERP title
    cleaned_title = title
    # Remove price/buy/review/location trailers more aggressively
    cleaned_title = re.sub(r"\s*[-|:•(]+\s*(?:Buy Online|Online|Store|Shop|Price|Best Price|Reviews?|Ratings?|Offers?|India|₹|Rs\.?|MRP|Uses|Side Effects|,).*", "", cleaned_title, flags=re.I)
    # Remove site names at start/end more robustly
    site_names_pattern = r"Amazon\.in|Flipkart\.com|1mg|Netmeds|Apollo Pharmacy|Pharmeasy|Medplusmart|Walmart\.com"
    cleaned_title = re.sub(rf"^\s*({site_names_pattern})[:\s-]*", "", cleaned_title, flags=re.I).strip()
    cleaned_title = re.sub(rf"\s*[-|:]\s*({site_names_pattern})$", "", cleaned_title, flags=re.I).strip()
    # Remove common generic phrases often appended
    cleaned_title = re.sub(r"\s*-\s*(?:Buy|Shop|Order|Get|View|Check|Find|Compare).*", "", cleaned_title, flags=re.I).strip()
    # Remove trailing quantity/pack indicators if they look separated " - 60 Tablets"
    cleaned_title = re.sub(r"\s*[-]\s*\d+\s*(?:'s| Tablets?| Capsules?| Strips?| ml| mg| gm| Pack\s*of\s*\d+)$", "", cleaned_title, flags=re.I).strip()
    # Remove parenthesis content if short and generic (like counts, simple descriptors)
    cleaned_title = re.sub(r"\s+\(\s*(?:\d+\s*(?:'s|pcs|nos?|tabs?|caps?|strips?)|Pack of \d+|[a-zA-Z\s]{1,15})\s*\)$", "", cleaned_title, flags=re.I).strip() # Expanded pattern slightly
    # General cleanup
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()

    # 2. Try finding a better title from H1 or specific product title elements on the page
    product_title_from_page = None
    # Common title elements (add more based on inspection - prioritize specific IDs/classes)
    selectors = [
        '#productTitle', # Amazon
        '.B_NuCI', '.VU-ZEz', '.yhB1nd', # Flipkart (examples, check classes)
        '.ProductTitle__product-title___', '.ProductHeader__title-content___', # 1mg (examples)
        '.prodName', '.product-title .prod-name', # Netmeds (examples)
        '.PdpWeb_productName__', '.style__pro-title___', # Apollo (examples)
        '.ProductCard__name___', '.ProductPage__name___', # Pharmeasy (examples)
        '.page-title span[data-ui-id="page-title-wrapper"]', '.product-name h1', # Medplus?
        'h1[itemprop="name"]', # Schema in H1
        'h1', # General H1 as fallback
        '[itemprop="name"]' # Schema.org (lower priority than H1 usually)
    ]
    for sel in selectors:
        element = soup.select_one(sel)
        if element:
            potential_title_raw = element.get_text(" ", strip=True)
            # Clean the potential title slightly (remove extra spaces)
            potential_title = re.sub(r'\s+', ' ', potential_title_raw).strip()
            # Basic validation: not too short, not too long, doesn't look like a generic heading or error message
            if potential_title and 5 < len(potential_title) < 200 and not any(w in potential_title.lower() for w in ["review", "customer", "specification", "description", "details", "results for", "not found", "error", "search", "category", "filter", "sort by"]):
                # Remove common prefixes like "Buy " often added in H1
                potential_title = re.sub(r"^(?:Buy|Order|Get|Shop)\s+", "", potential_title, flags=re.I).strip()
                product_title_from_page = potential_title
                logger.debug(f"Found potential title from page selector '{sel}': {product_title_from_page}")
                break # Take the first good one found, based on selector priority

    # 3. Decide which title to use
    final_title = cleaned_title # Default to cleaned SERP title
    if product_title_from_page:
        # Prefer page title if it's significantly different, longer, or seems more specific
        # Calculate similarity (simple word overlap heuristic)
        serp_words = set(final_title.lower().split())
        page_words = set(product_title_from_page.lower().split())
        overlap = len(serp_words & page_words)
        max_len = max(len(serp_words), len(page_words))
        similarity = overlap / max_len if max_len > 0 else 0

        # Use page title if:
        # - It's significantly longer (e.g., > 30% longer)
        # - OR it contains most of the SERP title words plus more (high overlap but page is longer)
        # - OR similarity is low (suggesting SERP title was truncated/generic)
        if (len(product_title_from_page) > len(final_title) * 1.3) or \
           (similarity > 0.6 and len(page_words) > len(serp_words)) or \
           (similarity < 0.5 and len(product_title_from_page) > 10): # Avoid very short dissimilar titles
             logger.info(f"Using page title: '{product_title_from_page}' over SERP title: '{final_title}' (Similarity: {similarity:.2f})")
             final_title = product_title_from_page
        else:
             logger.info(f"Keeping cleaned SERP title: '{final_title}' (Page title '{product_title_from_page}' too similar or not better)")


    # 4. Final cleanup and length limit
    final_title = re.sub(r'\s+', ' ', final_title).strip()
    # Limit length slightly less aggressively
    if len(final_title) > 180: # Apply final length limit
         # Try to cut at a word boundary
         last_space = final_title[:180].rfind(' ')
         if last_space > 100: # Ensure reasonable length after cut
             final_title = final_title[:last_space] + "..."
         else:
             final_title = final_title[:177] + "..." # Hard cut if no good space

    # Ensure title is not empty or a generic placeholder after all cleaning
    if not final_title or final_title.lower() in ["product", "details", "search results", "buy", "online", "shop", "category", "health", "medicine", "page", "item", "information"]:
        logger.warning(f"Final title deemed generic or unavailable: '{final_title}' from original '{title}'")
        return "Product Title Unavailable"

    return final_title

# --- Product Unavailable Check (Unchanged) ---
def product_unavailable(soup):
    """Checks for signs that the product is out of stock or unavailable."""
    text = soup.get_text(" ", strip=True).lower()
    keywords = [
        "currently unavailable", "out of stock", "sold out",
        "item is unavailable", "notify me when available", "back in stock soon",
        "product is not available", "discontinued", "no longer available",
        "item cannot be shipped", # Amazon specific
        "temporarily unavailable", "currently out of stock", # More variations
        "this item is currently out of stock"
    ]
    # Specific selectors (Need regular updates - prioritize ones clearly indicating OOS)
    unavailable_selectors = [
        "#availability .a-color-price:contains('unavailable')", # Amazon specific text check
        "#availability .a-color-error", # Amazon error color often used
        "#outOfStock", "[class*='outOfStock']", "[class*='OutOfStock']", # Common IDs/classes
        "._16FRp0", "._3_GKL7", # Flipkart 'Sold Out'/'Unavailable' classes (examples)
        ".AvailabilityMessage__not-available___", "[class*='AvailabilityNotifier']", # 1mg example classes
        ".text-unavailable", ".stock-info.unavailable", ".stock-indicator.out-of-stock",
        "button[disabled][title*='Out of Stock']", # Buttons indicating unavailability
        "button:contains('Notify Me')", "button:contains('Notify when available')", # Buttons
        ".product-stock-status.unavailable",
        ".oos-label", ".sold-out-label", # More label examples
        "p.stock.unavailable", "div.stock.unavailable"
    ]
    for sel in unavailable_selectors:
        # Use select for flexibility, limit to avoid overly broad selectors causing issues
        elements = soup.select(sel, limit=5)
        for element in elements:
             element_text = element.get_text(" ", strip=True).lower()
             # Check element text directly for keywords or specific phrases
             if any(kw in element_text for kw in ["unavailable", "out of stock", "sold out", "notify me", "discontinued"]):
                 logger.info(f"Product unavailable detected by selector '{sel}' text: '{element_text[:60]}...'")
                 return True

    # Check the broader page text content as a fallback - use with caution
    # Focus check around potential price/add-to-cart areas if possible
    cart_area = soup.select_one('#addToCart,#buyNow,#buybox,#centerCol,#rightCol') # Example areas near button/price
    area_text = cart_area.get_text(" ", strip=True).lower() if cart_area else text # Use focused area if found, else full text

    if any(keyword in area_text for keyword in keywords):
         # Avoid triggering on generic footer text etc. - check length maybe?
         if len(area_text) < len(text) * 0.5: # Check if the match is in a smaller section
             logger.info(f"Product unavailable detected by keyword in focused page area.")
             return True
         elif len(area_text) == len(text): # Fallback to full text check
             logger.info(f"Product unavailable detected by keyword in page text.")
             return True

    return False

# --- End of Helper Functions ---


# --- MedComparator Class (Unchanged logic, minor adjustments for reuse) ---
class MedComparator:
    def __init__(self):
        """Initializes the MedComparator."""
        self.session = None
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
             logger.warning("GROQ_API_KEY not found in environment variables. LLM features will be disabled.")
             self.llm = None
        else:
             try:
                 # Instantiate ChatGroq
                 self.llm = ChatGroq(
                     model_name="llama3-70b-8192", # Updated model name
                     groq_api_key=groq_api_key,
                     temperature=0.4,
                     max_tokens=250
                 )
                 logger.info("ChatGroq LLM initialized (llama3-70b-8192).")
             except ImportError:
                 logger.error("Failed to import ChatGroq. Make sure 'langchain-groq' is installed (`pip install langchain-groq`). LLM features disabled.")
                 self.llm = None
             except Exception as e:
                 logger.error(f"Failed to initialize ChatGroq LLM: {e}. LLM features disabled.")
                 self.llm = None

        self.cache = {} # Simple dict cache for SERP results
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Ch-Ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Connection": "keep-alive",
        }
        # Reset counters within context manager (__aenter__)

    async def __aenter__(self):
        """Initializes the async HTTP session and resets counters."""
        timeout = aiohttp.ClientTimeout(total=20, connect=8, sock_connect=8, sock_read=15)
        connector = aiohttp.TCPConnector(limit_per_host=3, force_close=True, enable_cleanup_closed=True)
        self.session = aiohttp.ClientSession(headers=self.headers, connector=connector, timeout=timeout)
        # Initialize counters here for each usage context
        self.domain_request_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
        self.total_scrapes_done = 0
        self.request_count = 0 # Also reset concurrent request counter
        logger.debug("MedComparator context entered, session created, counters reset.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the async HTTP session."""
        if self.session and not self.session.closed:
             await self.session.close()
             await asyncio.sleep(0.25)
        self.session = None
        logger.info(f"MedComparator context exited. Session closed. Total pages scraped in this context: {self.total_scrapes_done}")
        # Log exception info if one occurred
        if exc_type:
            logger.error(f"Exception occurred within MedComparator context: {exc_type.__name__}: {exc_val}")


    # --- Core Methods (search_products, filter_products, fetch_domain_results, process_serp_results, scrape_product_page) ---
    # These methods remain exactly the same as you provided them above.
    # They contain the logic for searching, filtering, rate limiting, and scraping.

    async def search_products(self, query):
        """Searches SERP for products across allowed domains and initiates scraping."""
        # Simple cache check based on the adjusted query
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        if cache_key in self.cache:
             logger.info(f"Using cached results for query: {query}")
             return self.cache[cache_key] # Return cached data if available

        logger.info(f"Searching SERP for adjusted query: {query}")
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
             logger.error("SERPER_API_KEY not found in environment variables. Cannot perform search.")
             return []

        # Prepare headers and tasks for Serper API calls
        serper_headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
        tasks = []
        for domain in ALLOWED_DOMAINS:
            params = {
                "q": f"{query} site:{domain}",
                "num": 5, # Request fewer results per domain initially (e.g., 5)
                "hl": "en",
                "gl": "in", # Geo-location India
                #"location": "Pune,Maharashtra,India" # More specific location if needed
            }
            tasks.append(self.fetch_domain_results(params, serper_headers))

        # Gather SERP results from all domains
        results_nested = await asyncio.gather(*tasks)
        # Flatten the list and ensure items are dictionaries
        all_products_raw = [p for sublist in results_nested for p in sublist if isinstance(p, dict)]
        logger.info(f"Received {len(all_products_raw)} raw organic results from SERP.")

        # Process SERP results to identify valid product links and initiate scraping
        processed_products = await self.process_serp_results(all_products_raw)

        # Final filtering and sorting of scraped product data
        filtered_products = self.filter_products(processed_products)

        # Store in cache before returning
        self.cache[cache_key] = filtered_products
        logger.info(f"Found {len(filtered_products)} relevant products after scraping and filtering for query: {query}")
        return filtered_products

    def filter_products(self, products):
        """Filters, deduplicates, and sorts the final list of scraped products."""
        if not products: return []

        filtered = []
        seen_links = set()
        seen_titles_sources = set()
        domain_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
        max_per_source = 2 # Reduced max products from the same source (e.g., 2 from Amazon)
        total_max = 8 # Reduced max total products to show

        logger.info(f"Filtering {len(products)} scraped products...")
        for p in products:
            # Basic validation
            if not p or not isinstance(p, dict): continue
            name = p.get("name", "Product Title Unavailable")
            link = p.get("link")
            source = p.get("source") # Source should be the simple domain name (e.g., Amazon)

            if not link or name == "Product Title Unavailable" or not source:
                 logger.debug(f"Skipping product due to missing essential fields: Name={name}, Link={link}, Source={source}")
                 continue

            # Normalize link for seen check (remove query params, fragment)
            parsed_link = urlparse(link)
            normalized_link = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
            source_domain = extract_domain(link) # Get consistent domain for counting

            # Use a combination of simplified title and source for uniqueness check
            simplified_title = ''.join(filter(str.isalnum, name.lower()))[:50] # Use first 50 chars
            title_source_key = f"{simplified_title}-{source.lower()}"

            # --- Deduplication and Limits ---
            if normalized_link in seen_links:
                 logger.debug(f"Skipping duplicate normalized link: {normalized_link} (Original: {link})")
                 continue
            if title_source_key in seen_titles_sources:
                 logger.debug(f"Skipping duplicate simplified name/source: {title_source_key} (Name: {name})")
                 continue
            # Domain counting needs to use the extracted domain, not the title-cased source
            if source_domain and domain_counts.get(source_domain, 0) >= max_per_source:
                 logger.debug(f"Skipping due to source limit ({max_per_source}) for {source_domain}: {name}")
                 continue
            if len(filtered) >= total_max:
                 logger.info(f"Reached total product limit ({total_max}). Stopping filter.")
                 break

            # --- Additional Quality Checks ---
            # Skip if very little info was scraped AND no certifications/details
            is_low_info = (p.get('price', 'Price not available') == "Price not available" and
                           p.get('rating', 'Rating not available') == "Rating not available" and
                           not p.get('details') and # Check if details list is empty
                           not p.get('certifications')) # Check if certifications list is empty

            if is_low_info:
                 logger.info(f"Skipping low-information product: {name} (Link: {link})")
                 continue

            # Skip if title is overly generic even after cleaning (fallback check)
            if name.lower() in ["product", "item", "health care", "medicine", "supplement"]:
                 logger.info(f"Skipping product with overly generic final title: {name} (Link: {link})")
                 continue


            # --- Add to Filtered List ---
            filtered.append(p)
            seen_links.add(normalized_link)
            seen_titles_sources.add(title_source_key)
            if source_domain: # Only increment if domain was extracted
                 domain_counts[source_domain] = domain_counts.get(source_domain, 0) + 1

        # Sort by price (ascending, placing 'not available' last)
        def sort_key_price(item):
            price_str = item.get('price', 'Price not available')
            if price_str == 'Price not available':
                return float('inf')
            try:
                # Use Decimal for sorting as well
                cleaned_val = re.sub(r"[^0-9.]", "", price_str)
                return Decimal(cleaned_val) if cleaned_val else float('inf')
            except Exception:
                return float('inf')

        filtered.sort(key=sort_key_price)
        logger.info(f"Finished filtering. Returning {len(filtered)} products.")
        return filtered

    async def fetch_domain_results(self, params, headers):
        """Fetches search results for a specific domain from Serper API."""
        domain = params['q'].split("site:")[-1]
        logger.info(f"Fetching SERP for {domain}...")
        try:
            if not self.session or self.session.closed:
                 logger.error(f"Attempted SERP fetch with closed session for {domain}.")
                 return [] # Cannot proceed without session

            # Use POST as recommended by Serper
            async with self.session.post("https://google.serper.dev/search", json=params, headers=headers) as res:
                if res.status == 200:
                    try:
                         results = await res.json()
                         organic_results = results.get("organic", [])
                         logger.info(f"Got {len(organic_results)} results from SERP for {domain}")
                         return organic_results
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                         logger.error(f"Failed to decode/parse JSON response from Serper for {domain}: {e} - Response: {await res.text()}")
                         return []
                else:
                    # Log error with status and response text
                    response_text = await res.text()
                    logger.error(f"SERP API request failed for {domain} with status {res.status}: {response_text[:500]}") # Log first 500 chars
                    return []
        except aiohttp.ClientConnectorCertificateError as e:
             logger.error(f"SSL Certificate error during SERP API request for {domain}: {e}. Check system certificates or network proxy.")
             return []
        except aiohttp.ClientConnectorError as e:
             logger.error(f"Connection error during SERP API request for {domain}: {e}")
             return []
        except asyncio.TimeoutError:
             logger.error(f"Timeout during SERP API request for {domain}")
             return []
        except aiohttp.ClientError as e: # Catch other client errors
             logger.error(f"Client network error during SERP API request for {domain}: {e}")
             return []
        except Exception as e:
             logger.error(f"Unexpected error during SERP API request for {domain}: {e}", exc_info=True)
             return []

    async def process_serp_results(self, organic_results):
        """Filters SERP results and creates scraping tasks for valid product pages."""
        tasks = []
        valid_links_scheduled = set()
        if not organic_results:
             logger.warning("No organic results received from SERP API to process.")
             return []

        # Define limits here, using attributes set in __aenter__
        max_total_scrapes = 15 # Hardcoded limit (could be made configurable)
        max_requests_per_domain = 6 # Hardcoded limit

        logger.info(f"Processing {len(organic_results)} raw organic results for scraping (Total Limit: {max_total_scrapes}, Per Domain: {max_requests_per_domain})...")

        for result in organic_results:
            if self.total_scrapes_done >= max_total_scrapes:
                 logger.info(f"Reached maximum total scrapes limit ({max_total_scrapes}). Stopping scheduling.")
                 break

            if not isinstance(result, dict): continue # Skip non-dict results
            link = result.get("link")
            title = result.get("title", "")
            if not link or not title:
                 logger.debug(f"Skipping SERP result with missing link or title: {result}")
                 continue

            # Normalize link for checking duplicates
            parsed_link = urlparse(link)
            normalized_link = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"

            # Check if URL is allowed, looks like a product page, and hasn't been scheduled
            source_domain = extract_domain(link)
            if (is_allowed_url(link) and
                not is_generic_result(title, link) and
                normalized_link not in valid_links_scheduled):

                # Check domain request limit before scheduling
                if source_domain and self.domain_request_counts.get(source_domain, 0) < max_requests_per_domain:
                     # Check total scrapes limit
                     if self.total_scrapes_done < max_total_scrapes:
                         self.domain_request_counts[source_domain] = self.domain_request_counts.get(source_domain, 0) + 1
                         self.total_scrapes_done += 1 # Increment total counter
                         # Pass self (the comparator instance) to the scrape method
                         tasks.append(self.scrape_product_page(result))
                         valid_links_scheduled.add(normalized_link)
                         logger.debug(f"Scheduled scraping for: {link} (Domain: {source_domain}, Total Scrapes: {self.total_scrapes_done})")
                     else:
                         logger.info(f"Skipping scheduling for {link} due to total scrapes limit.")
                         # Break here might be too aggressive, let loop finish checking other domains
                elif not source_domain:
                     logger.warning(f"Could not extract domain for link, skipping: {link}")
                else: # Domain limit reached
                     logger.info(f"Skipping scheduling for {link} due to domain request limit for {source_domain} ({self.domain_request_counts.get(source_domain, 0)}/{max_requests_per_domain}).")
            # Log reasons for skipping
            elif normalized_link in valid_links_scheduled: pass # Already logged as scheduled or seen
            elif not is_allowed_url(link): logger.debug(f"Skipping disallowed domain: {link}")
            else: logger.debug(f"Skipping generic result: {title} | {link}")


        if not tasks:
             logger.warning("No valid, non-generic product links found to scrape after filtering SERP results.")
             return []

        logger.info(f"Gathering results for {len(tasks)} scraping tasks...")
        # Use asyncio.gather with return_exceptions=True to handle individual task failures
        scraped_results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Finished gathering scrape results (incl. potential errors).")

        # Process results, filtering out exceptions and None values
        final_products = []
        for i, res in enumerate(scraped_results):
             # Try to get the original link associated with the task for better logging
             original_link = "Unknown Link"
             task = tasks[i]
             # Accessing the link stored in the scrape_product_page method instance if possible
             # This assumes scrape_product_page is bound to the MedComparator instance
             if hasattr(task, '__func__') and task.__func__.__name__ == 'scrape_product_page' and hasattr(task, '__self__'):
                # The 'result' dict passed to scrape_product_page might be accessible via closure or args if needed
                # A simpler way might be to store the link on the instance temporarily during the call
                # Let's try accessing result directly if it was stored (needs adjustment in scrape_product_page)
                # Safest: just log that *a* task failed if we can't easily get the link
                pass # Cannot reliably get link back here without modifying scrape_product_page structure


             if isinstance(res, Exception):
                 logger.error(f"Scraping task failed with exception: {res}")
             elif res is None:
                 logger.warning(f"Scraping task returned None (likely filtered or failed).")
             elif isinstance(res, dict):
                 final_products.append(res)
             else:
                 logger.error(f"Scraping task returned unexpected type: {type(res)}")

        return final_products


    async def scrape_product_page(self, result):
        """Scrapes a single product page for details, price, rating, etc."""
        link = result["link"]
        serp_title = result["title"]
        # Store link temporarily on instance ONLY FOR THIS CALL's logging purposes (might cause issues if class is used concurrently elsewhere)
        # A better approach is passing logger or context, but this is minimal change
        self._current_scrape_link = link # Temporary storage for logging
        logger.info(f"Scraping: {link}")
        current_attempt = 0
        max_attempts = 2 # Retry once on specific failures
        base_delay = 0.6

        while current_attempt < max_attempts:
            current_attempt += 1
            product_data = None # Reset product data for retry
            try:
                # Calculate sleep duration with backoff and jitter
                sleep_duration = base_delay * (current_attempt ** 1.5) + (self.request_count * 0.03) + (0.2 * (hash(link) % 5)) # Exponential backoff + load factor + jitter
                logger.debug(f"Attempt {current_attempt}/{max_attempts}: Waiting {sleep_duration:.2f}s before scraping {link}")
                await asyncio.sleep(sleep_duration)
                self.request_count += 1 # Increment concurrent request counter

                # Ensure session exists and is not closed
                if not self.session or self.session.closed:
                     logger.error(f"Attempt {current_attempt}: Scraping attempted with closed/missing session for {link}.")
                     return None # Cannot proceed

                # Make the GET request
                async with self.session.get(link, allow_redirects=True) as res: # Timeout is set on session level
                    logger.debug(f"Attempt {current_attempt}: Received status {res.status} for {link}")
                    # Check for non-200 status codes
                    if res.status != 200:
                        log_func = logger.warning if res.status in [404, 410] else logger.error
                        log_func(f"Attempt {current_attempt}: Failed to fetch {link} - Status: {res.status}")
                        # Decide whether to retry based on status code
                        # Retry on server errors (5xx) or rate limiting (429), possibly 403 (permission denied)
                        if res.status in [403, 429, 500, 502, 503, 504] and current_attempt < max_attempts:
                            logger.info(f"Retrying ({current_attempt}/{max_attempts}) after status {res.status}...")
                            continue # Go to next attempt
                        else:
                             return None # Don't retry other errors or if max attempts reached

                    # Process successful response
                    html = None
                    try:
                        html_bytes = await res.read()
                        # Try decoding with multiple common encodings
                        try: html = html_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            try: html = html_bytes.decode('iso-8859-1')
                            except UnicodeDecodeError: html = html_bytes.decode('cp1252', errors='ignore') # Fallback
                        logger.debug(f"Attempt {current_attempt}: Successfully read/decoded HTML ({len(html)} chars) from {link}")
                    except Exception as e:
                        logger.error(f"Attempt {current_attempt}: Failed to read/decode HTML from {link}: {e}")
                        if current_attempt < max_attempts: continue
                        else: return None

                    if not html or len(html) < 500: # Check if HTML seems too small/empty
                         logger.warning(f"Attempt {current_attempt}: Empty or very small HTML content received from {link}")
                         if current_attempt < max_attempts: continue
                         else: return None

                    # --- Start Parsing and Extraction ---
                    soup = BeautifulSoup(html, "lxml") # Use lxml for speed

                    if product_unavailable(soup):
                         logger.info(f"Product detected as unavailable, skipping scrape: {link}")
                         return None # No point scraping unavailable products

                    domain = extract_domain(link)
                    name = clean_title(serp_title, soup)

                    if name == "Product Title Unavailable":
                         logger.warning(f"Could not extract a valid title for {link}, skipping.")
                         return None # Skip if no usable title

                    # --- Extraction Calls ---
                    price = extract_price(soup)
                    rating = extract_rating(soup, domain)
                    details = extract_product_details(soup, domain)
                    certifications = extract_certifications(soup)
                    # --- End Extraction ---


                    # Final check: ensure essential info exists OR details/certs present
                    if price == "Price not available" and rating == "Rating not available" and not details and not certifications:
                         logger.info(f"Skipping product due to lack of compelling info (price, rating, details, certs): {name} ({link})")
                         return None

                    # Determine source name (Capitalize first letter)
                    source_name = domain.split('.')[0].title() if '.' in domain else domain.title()
                    # Specific overrides if needed
                    if source_name == "Apollopharmacy": source_name = "Apollo Pharmacy"
                    elif source_name == "1mg": source_name = "1mg" # Keep as is
                    elif source_name == "Medplusmart": source_name = "MedPlus Mart"

                    product_data = {
                        "name": name,
                        "source": source_name,
                        "price": price,
                        "rating": rating,
                        "details": details, # This is now the refined list
                        "certifications": certifications, # This is the refined list
                        "link": link,
                    }
                    logger.info(f"Successfully scraped: {name} ({product_data.get('price', 'N/A')}, {product_data.get('rating', 'N/A')}, Details: {len(details)}, Certs: {len(certifications)})")
                    return product_data # Success, exit retry loop

            # --- Error Handling for the Scraping Attempt ---
            except asyncio.TimeoutError:
                 logger.warning(f"Attempt {current_attempt}: Timeout scraping {link}")
                 if current_attempt == max_attempts: return None
            except aiohttp.ClientSSLError as e:
                 logger.error(f"Attempt {current_attempt}: SSL Error scraping {link}: {e}. Check certificates or network. Aborting for this URL.")
                 return None # Don't retry SSL errors usually
            except aiohttp.ClientConnectionError as e: # Covers more connection issues
                 logger.warning(f"Attempt {current_attempt}: Connection error scraping {link}: {e}")
                 if current_attempt == max_attempts: return None # Only return None after last attempt
            except aiohttp.ClientPayloadError as e:
                 logger.warning(f"Attempt {current_attempt}: Payload error (incomplete read?) scraping {link}: {e}")
                 if current_attempt == max_attempts: return None
            except aiohttp.ClientResponseError as e: # Catch specific HTTP errors if needed
                 logger.warning(f"Attempt {current_attempt}: Client response error scraping {link}: Status={e.status}, Message={e.message}")
                 if current_attempt == max_attempts: return None
            except aiohttp.ClientError as e: # Catch other general client errors
                 logger.warning(f"Attempt {current_attempt}: Client network error scraping {link}: {e}")
                 if current_attempt == max_attempts: return None
            except Exception as e:
                 logger.error(f"Attempt {current_attempt}: Unexpected error scraping {link}: {e}", exc_info=True) # Log traceback for unexpected
                 return None # Don't retry unexpected errors generally
            finally:
                 # Decrement concurrent request counter safely
                 self.request_count = max(0, self.request_count - 1)
                 # Clean up temporary link storage
                 if hasattr(self, '_current_scrape_link'):
                     delattr(self, '_current_scrape_link')


        # If loop finishes without returning success
        logger.error(f"Scraping ultimately failed for {link} after {max_attempts} attempts.")
        return None

# --- New Core Function ---
async def get_comparison_data_async(query: str) -> list:
    """
    Fetches and scrapes product comparison data for the given query. (Async)

    This is the primary function to call programmatically for integration.

    Args:
        query: The user's search query (e.g., "Vitamin C 500mg tablets").

    Returns:
        A list of dictionaries, where each dictionary represents a product
        with its scraped details (name, source, price, rating, details, etc.).
        Returns an empty list if no products are found or an error occurs.
        The list is suitable for direct use or JSON serialization.
    """
    products = []
    # Ensure API keys are available before proceeding
    if not os.getenv("SERPER_API_KEY"):
        logger.error("SERPER_API_KEY is not set. Cannot perform search.")
        return []

    try:
        # Initialize comparator within the function scope for each call
        # This ensures fresh state (counters, session) for each query
        comparator = MedComparator()
        async with comparator: # Use the context manager for session setup/teardown
            logger.info(f"Starting async comparison data fetch for query: '{query}'")
            adjusted_query = adjust_query(query) # Use existing adjustment logic
            logger.info(f"Adjusted query for search: '{adjusted_query}'")
            products = await comparator.search_products(adjusted_query) # Call the core search method

            if not products:
                logger.warning(f"No specific, available products found for '{query}' via search_products.")
            else:
                logger.info(f"Successfully fetched {len(products)} products for query '{query}'.")

    except aiohttp.ClientError as e:
        logger.error(f"A network error occurred during get_comparison_data_async for query '{query}': {e}", exc_info=True)
        products = [] # Ensure empty list is returned on network errors
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_comparison_data_async for query '{query}': {e}", exc_info=True)
        products = [] # Ensure empty list is returned on other errors
    # Context manager (__aexit__) handles session closing

    return products # Return the list of product dictionaries


# --- Leader's Requested Function (Synchronous Wrapper) ---
def final_result(query: str) -> str:
    """
    Synchronous wrapper to get comparison data and return it as a JSON string.

    This function matches the format requested by the leader.
    It runs the asynchronous data fetching logic internally.

    Args:
        query: The user's search query.

    Returns:
        A JSON string representing the list of products, or '[]' on error or if no products found.
    """
    logger.info(f"Executing synchronous wrapper 'final_result' for query: {query}")
    json_output = "[]" # Default to empty JSON list string
    try:
        # nest_asyncio.apply() should already be called globally at the start
        # Run the async function to get the product list
        product_list = asyncio.run(get_comparison_data_async(query))

        # Serialize the result list to JSON string
        # Use ensure_ascii=False to handle symbols like ₹ correctly
        # Use default handler for non-serializable objects (though data should be fine)
        json_output = json.dumps(product_list, indent=2, ensure_ascii=False, default=lambda o: '<not serializable>')

    except RuntimeError as e:
         # Handle cases where asyncio.run cannot be used (e.g., nested loops without nest_asyncio properly configured)
         logger.error(f"RuntimeError calling asyncio.run in final_result (maybe nested loops?): {e}")
         json_output = "[]" # Return empty JSON list on error
    except Exception as e:
        logger.error(f"Unexpected error in final_result wrapper for query '{query}': {e}", exc_info=True)
        json_output = "[]" # Return empty JSON list on error

    logger.info(f"final_result returning JSON (length: {len(json_output)}) for query: {query}")
    return json_output


# --- Interactive Mode Function (for direct execution/testing) ---
async def main_interactive():
    """Interactive main function to demonstrate usage when script is run directly."""

    # Get user input from terminal
    try:
        query = input("Enter medicine/supplement name or health concern (e.g., 'Vitamin C 500mg tablets'): ").strip()
        if not query:
             print("No query entered. Exiting.")
             return
    except EOFError:
         print("\nNo input received (EOF). Exiting.")
         return

    print(f"\nFetching data for: {query}...")
    # Call the reusable async function
    products = await get_comparison_data_async(query) # Now we call the reusable function

    # --- Output Section (similar to original main) ---
    if not products:
         print(f"\nNo specific, available products found matching your query '{query}'. Try refining your search.", file=sys.stderr)
         print("\n[]") # Output empty JSON list
         return

    # 1. ALWAYS print the full JSON data first
    print("\n--- Product List (JSON returned by get_comparison_data_async) ---")
    try:
        # Use ensure_ascii=False to print Rupee symbol etc. directly if terminal supports UTF-8
        print(json.dumps(products, indent=2, ensure_ascii=False))
    except TypeError as e:
         logger.error(f"Failed to serialize results to JSON: {e}. Likely non-serializable data.")
         # Attempt to serialize safely, replacing errors
         print(json.dumps(products, indent=2, ensure_ascii=False, default=lambda o: '<not serializable>'))
    except Exception as e: # Catch potential encoding errors on print
         logger.error(f"Failed to print JSON output (ensure_ascii=False): {e}. Trying with ensure_ascii=True.")
         try:
             print(json.dumps(products, indent=2, ensure_ascii=True))
         except Exception as final_e:
             logger.error(f"Final attempt to print JSON failed: {final_e}")
             print("[]") # Final fallback

    # 2. Optionally proceed with Table and AI Analysis
    groq_key_exists = bool(os.getenv("GROQ_API_KEY"))
    if groq_key_exists: # Only if LLM is available
        analysis_requested = False
        try:
            # Ask user if they want analysis
            while True:
                 analyze_input = input(f"\nFound {len(products)} products. Generate comparison table & AI recommendation? (y/n): ").lower().strip()
                 if analyze_input == 'y':
                     analysis_requested = True
                     break
                 elif analyze_input == 'n':
                     analysis_requested = False
                     break
                 else:
                     print("Please enter 'y' or 'n'.")
        except EOFError:
             print("\nInput interrupted. Skipping analysis.")
             analysis_requested = False

        if analysis_requested:
            # --- Display Comparison Table ---
            print("\n--- Product Comparison Table ---")
            if TABULATE_AVAILABLE:
                table_data = []
                headers = ["#", "Name", "Source", "Price", "Rating"] # Simplified headers
                for i, p in enumerate(products):
                     # Truncate long names for table display
                     name_display = p.get('name', 'N/A')
                     if len(name_display) > 55: # Slightly shorter for table
                         name_display = name_display[:52] + "..."
                     table_data.append([
                         i + 1,
                         name_display,
                         p.get('source', 'N/A'),
                         p.get('price', 'N/A'),
                         p.get('rating', 'N/A')
                     ])
                # Use tabulate for formatted output
                try:
                     print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[None, 55, None, None, None])) # 'grid' format looks nice
                except Exception as e:
                     logger.error(f"Error formatting table with tabulate: {e}")
                     print("(Error displaying formatted table - tabulate library issue)")
            else:
                print("(Install 'tabulate' library for a formatted table: pip install tabulate)")
                # Basic fallback table
                print(" | ".join(["#", "Name", "Source", "Price", "Rating"]))
                print("-" * 80) # Adjusted separator width
                for i, p in enumerate(products):
                     name_display = p.get('name', 'N/A')
                     if len(name_display) > 30: name_display = name_display[:27]+"..."
                     print(f"{i+1:<2} | {name_display:<30} | {p.get('source', 'N/A'):<15} | {p.get('price', 'N/A'):<15} | {p.get('rating', 'N/A'):<10}")


            # --- Generate LLM Analysis ---
            print("\n--- Generating AI Recommendation (takes a moment)... ---")
            try:
                # Initialize LLM instance specifically for analysis here
                llm_instance = ChatGroq(
                    model_name="llama3-70b-8192",
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.4,
                    max_tokens=250
                )
                logger.info("Temporary LLM instance created for analysis.")

                # Prepare data for LLM - Send core info + maybe 1-2 key details if available
                llm_analysis_data = []
                for p in products:
                     item_data = {
                         "Title": p["name"],
                         "Price": p["price"],
                         "Rating": p["rating"],
                         "Source": p.get("source", "N/A"),
                         # Add first detail if present and seems informative
                         "Highlight": p['details'][0] if p.get('details') else "N/A"
                     }
                     llm_analysis_data.append(item_data)


                if not llm_analysis_data:
                     print("Error: Not enough product data to generate analysis.")
                else:
                     # Create a more readable string format for the prompt
                     prompt_product_lines = []
                     for i, p in enumerate(llm_analysis_data):
                         line = f"{i+1}. {p['Title']} (Source: {p['Source']}, Price: {p['Price']}, Rating: {p['Rating']})"
                         if p['Highlight'] != "N/A":
                             line += f" - Highlight: '{p['Highlight'][:100]}...'" # Truncate highlight
                         prompt_product_lines.append(line)
                     prompt_data_string = "\n".join(prompt_product_lines)

                     user_query_context = query # Use original user query for context
                     prompt = f"""You are an AI shopping assistant providing a concise recommendation based on scraped product data for a user searching for '{user_query_context}'.

Here is the product data found:
{prompt_data_string}

Instructions:
1.  Analyze the products based ONLY on the provided Title, Source, Price, Rating, and Highlight.
2.  Directly recommend **1 or 2 specific products** from the list (referencing their number, e.g., "Product #1") that appear most suitable or offer good value for '{user_query_context}'. Consider price, rating, and potential relevance from the title/highlight.
3.  Briefly justify your recommendation(s) (1-2 sentences max per product), mentioning the key factors like price, rating, or a relevant highlight.
4.  If prices are very different, acknowledge the trade-off (e.g., "Product #X is cheaper but has lower rating/less info").
5.  Keep the *entire response* concise (around 3-5 sentences total). Start directly with the recommendation.
6.  Do NOT invent features or benefits not present in the data. Do NOT give medical advice.

Concise Recommendation:
"""
                     logger.info("Sending prompt to LLM for analysis...")
                     response = llm_instance.invoke(prompt)
                     analysis_text = response.content.strip() if response and hasattr(response, 'content') else "No response content."

                     # Basic cleanup of LLM output
                     analysis_text = re.sub(r"^(Concise Recommendation:\s*)+", "", analysis_text, flags=re.IGNORECASE).strip()

                     print("\n--- AI Recommendation ---")
                     print(analysis_text)
                     print("\n**IMPORTANT:** This AI recommendation is based *only* on the limited data shown (Title, Price, Rating, Highlight). It is NOT medical advice. Product details, availability, and prices change frequently. Always verify information on the seller's website and consult a healthcare professional for health decisions.")
                     print("------------------------")

            except Exception as e:
                logger.error(f"LLM analysis failed: {e}", exc_info=True)
                # Check for specific API errors if possible (e.g., Groq rate limits, auth)
                if "400" in str(e) and "model_decommissioned" in str(e):
                     print("\nError: The AI model used for analysis is currently unavailable. Please update the script or try again later.")
                elif "429" in str(e):
                     print("\nError: AI analysis request failed due to rate limiting. Please try again later.")
                else:
                     print("\nError: Could not generate AI analysis at this time.")
    else: # LLM key not available case
         logger.info("GROQ_API_KEY not set, skipping analysis option in interactive mode.")


# --- Main Execution Block ---
# if __name__ == "__main__":
    # # --- Pre-run Checks ---
    # # Check for API key existence early
    # if not os.getenv("SERPER_API_KEY"):
    #     logger.error("Fatal Error: SERPER_API_KEY environment variable is not set. The application cannot search for products.")
    #     print("Fatal Error: SERPER_API_KEY environment variable is not set. Please create a .env file with your key.", file=sys.stderr)
    #     sys.exit(1) # Exit with error code
    # if not os.getenv("GROQ_API_KEY"):
    #      # Log warning but allow continuation without LLM features
    #      logger.warning("GROQ_API_KEY environment variable is not set. AI analysis features will be disabled.")
    #      print("Warning: GROQ_API_KEY not found. AI analysis will be unavailable.", file=sys.stderr)

    # # --- Run Mode ---
    # # This block now only runs when the script is executed directly.
    # # It runs the interactive demo by default.

    # print("Running MedComparator script in interactive mode...")

    # try:
    #     # On Windows, default event loop policy might cause issues with aiohttp+asyncio
    #     if sys.platform == 'win32':
    #          try:
    #              # Check if policy is already set or if ProactorEventLoop is default
    #              if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
    #                  # Try setting Selector policy only if needed, avoid errors if already set or not applicable
    #                  current_loop = asyncio.get_event_loop()
    #                  if not isinstance(current_loop, asyncio.SelectorEventLoop):
    #                      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    #                      logger.info("Set WindowsSelectorEventLoopPolicy for interactive mode.")
    #          except Exception as policy_e:
    #               logger.warning(f"Could not set WindowsSelectorEventLoopPolicy (may not be needed): {policy_e}")

    #     # Run the async interactive function
    #     asyncio.run(main_interactive())

    # except KeyboardInterrupt:
    #      print("\nSearch interrupted by user. Exiting.")
    #      logger.info("Interactive process interrupted by user (KeyboardInterrupt).")
    # except Exception as e:
    #      logger.error("Application crashed unexpectedly in __main__ interactive run", exc_info=True)
    #      print(f"\nAn unexpected error occurred during interactive run: {e}", file=sys.stderr)
    #      sys.exit(1)
    # finally:
    #      logger.info("Script finished interactive execution.")

    # # --- Example showing how 'final_result' could be called (commented out by default) ---
    # # print("\n--- Example: Calling final_result('Vitamin D3') ---")
    # # example_query = "Vitamin D3 60000 IU"
    # # result_json_string = final_result(example_query)
    # # print(f"Received JSON String (first 500 chars):\n{result_json_string[:500]}...")
    # # print("-------------------------------------------------")

    # print(final_result("Vitamin C 500mg tablets"))