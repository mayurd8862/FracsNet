# import os
# import re
# import json
# import aiohttp
# import asyncio
# import streamlit as st
# import pandas as pd
# from bs4 import BeautifulSoup
# from llama_cpp import Llama
# from dotenv import load_dotenv
# import hashlib
# import nest_asyncio
# from urllib.parse import urlparse
# from textblob import TextBlob
# import logging

# nest_asyncio.apply()
# load_dotenv()
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ALLOWED_DOMAINS = [
#     "amazon.in", "flipkart.com", "1mg.com",
#     "netmeds.com", "apollopharmacy.in", "pharmeasy.in",
#     "medplusmart.com", "walmart.com" # Note: Apollo changed domain, Pharmeasy uses .in
# ]

# def correct_spelling(text):
#     try:
#         return str(TextBlob(text).correct())
#     except Exception as e:
#         logging.warning(f"Spelling correction failed: {e}")
#         return text

# def adjust_query(user_query):
#     corrected = correct_spelling(user_query)
#     base = re.sub(r"\b(compare|vs|versus|suggest me \d+|show me|which is better|that are there|find|get me)\b", "", corrected, flags=re.IGNORECASE)
#     base = re.sub(r"\s+", " ", base).strip()
#     # Avoid adding "buy online india" if domain context is already present or implied
#     if not any(domain in base for domain in ["amazon", "flipkart", "1mg", "netmeds", "apollo", "pharmeasy", "medplus"]):
#          base = f"{base} buy online india"
#     return base.strip()

# def extract_domain(url):
#     try:
#         parsed = urlparse(url)
#         domain = parsed.netloc.lower().replace('www.', '').replace('m.', '').split(':')[0]
#         # Handle potential domain variations like apollo.pharmacy vs apollopharmacy.in
#         if domain == "apollo.pharmacy":
#              return "apollopharmacy.in"
#         return domain
#     except Exception:
#         return ""

# def is_allowed_url(url):
#     domain = extract_domain(url)
#     return any(allowed == domain for allowed in ALLOWED_DOMAINS)

# def is_generic_result(title, url):
#     generic_terms = [
#         "bestsellers", "product price list", "shop online", "trusted online",
#         "order medicine", "category", "all products", "health conditions",
#         "store near you", "pharmacy online", "medicine list"
#     ]
#     combined = (title + " " + url).lower()
#     # Check if the title itself is too generic
#     if title.lower() in ["medicines", "health products", "online pharmacy", "supplements"]:
#         return True
#     # Check for generic path segments
#     if any(f"/{term}/" in url.lower() for term in ["category", "categories", "all", "list"]):
#          return True
#     return any(term in combined for term in generic_terms)

# def is_valid_detail(sentence):
#     sentence = sentence.strip()
#     if len(sentence) < 25 or len(sentence) > 300: # Adjusted length limits
#         return False

#     stop_phrases = [
#         "privacy policy", "terms of use", "return policy", "add to cart", "in stock",
#         "buy now", "tell us about", "lower price", "csrf", "added to cart", "shipping information",
#         "customer reviews", "related products", "login", "register", "track order", "specifications",
#         "highlights", "limited time offer", "best price guaranteed", "shop now", "free delivery",
#         "secure transaction", "go back", "view details", "learn more", "click here",
#         "frequently asked questions", "faq", "contact us", "about us", "need help",
#         "powered by", "copyright", "all rights reserved", "sign in", "your account",
#         "compare prices", "check availability", "select options", "quantity",
#         "description", "features", "key benefits", # Avoid section headers as details
#         "how to use", "safety information", "ingredients", "directions for use",
#         "subscribe", "newsletter", "download app", "visit store"
#     ]

#     # Check for exact or near-exact matches with stop phrases
#     lower_sentence = sentence.lower()
#     if lower_sentence in stop_phrases:
#         return False
#     if any(phrase in lower_sentence for phrase in stop_phrases):
#         return False

#     # Check for list markers or excessive symbols often found in non-prose content
#     if re.match(r"^\W*(\*|\-|•|>|\d+\.)\s+", sentence): # Starts with list markers
#         return False
#     if sentence.count('{') > 0 or sentence.count('}') > 0 or sentence.count('[') > 1 or sentence.count(']') > 1:
#         return False
#     if sentence.count('/') > 3 or sentence.count(':') > 3: # Avoid URL-like strings or code snippets
#         return False

#     # Check for overly promotional or generic website language
#     if re.search(r"(?:best|top|great|amazing|fantastic)\s+(?:deals?|offers?|price|quality)", lower_sentence):
#         return False
#     if re.search(r"shop|buy|order\s+now", lower_sentence):
#         return False
#     if "click" in lower_sentence and "here" in lower_sentence:
#         return False

#     # Basic check for sentence structure (presence of a verb might help) - simple heuristic
#     if len(lower_sentence.split()) > 3 and not any(verb in lower_sentence for verb in ["is", "are", "helps", "supports", "provides", "contains", "improves", "reduces", "promotes"]):
#          # Allow if it seems descriptive despite lacking common verbs
#          pass # Relaxing this check slightly as it can be too restrictive

#     return True

# def refine_details(details):
#     refined = []
#     for d in details:
#         # Clean up common residues
#         d = d.replace("View more", "").replace("Read more", "").strip()
#         d = re.sub(r'\s+', ' ', d) # Normalize whitespace
#         if d and is_valid_detail(d):
#              refined.append(d)

#     seen = set()
#     final = []
#     for d in refined:
#         # Use a simplified version for checking duplicates (lowercase, no punctuation)
#         simplified_d = ''.join(filter(str.isalnum, d.lower()))
#         if simplified_d and simplified_d not in seen:
#             seen.add(simplified_d)
#             final.append(d)

#     # Sort by relevance (heuristic: longer details might be more informative)
#     final.sort(key=len, reverse=True)
#     return final

# def extract_product_details(soup, domain):
#     details = []

#     # 1. Try ld+json description extraction (High Priority)
#     try:
#         for script in soup.find_all("script", type="application/ld+json"):
#             try:
#                 data = json.loads(script.string)
#                 if isinstance(data, list):
#                     # Find the Product object, could be nested
#                     product_data = None
#                     queue = data[:]
#                     while queue:
#                         item = queue.pop(0)
#                         if isinstance(item, dict):
#                             if item.get("@type") == "Product":
#                                 product_data = item
#                                 break
#                             # Check graph structure
#                             if "@graph" in item and isinstance(item["@graph"], list):
#                                 queue.extend(item["@graph"])
#                         elif isinstance(item, list): # Handle nested lists
#                             queue.extend(item)

#                 elif isinstance(data, dict) and data.get("@type") == "Product":
#                      product_data = data
#                 else:
#                     product_data = None


#                 if product_data and (desc := product_data.get("description")):
#                     # Split into sentences more carefully
#                     sentences = re.split(r'(?<=[.!?])\s+', desc)
#                     for sentence in sentences:
#                         sentence = sentence.strip()
#                         if sentence:
#                             details.append(sentence)
#                     if details: break # Stop if found via ld+json

#             except (json.JSONDecodeError, TypeError, AttributeError) as e:
#                  logging.debug(f"LD+JSON parsing error: {e}")
#                  continue
#     except Exception as e:
#         logging.error(f"Error processing script tags: {e}")


#     # 2. Domain-specific extraction (Medium Priority)
#     if len(details) < 3: # Only proceed if ld+json didn't yield enough
#         specific_details = []
#         try:
#             if "amazon.in" in domain:
#                 # Feature bullets
#                 bullets_div = soup.find("div", id="feature-bullets")
#                 if bullets_div:
#                     for li in bullets_div.find_all("li"):
#                         text = li.get_text(" ", strip=True)
#                         if text and not text.lower().startswith("see more product details"):
#                             specific_details.append(text)
#                 # Product description section
#                 desc_div = soup.find("div", id="productDescription")
#                 if desc_div:
#                      paragraphs = desc_div.find_all("p")
#                      if paragraphs:
#                           for p in paragraphs:
#                                specific_details.append(p.get_text(" ", strip=True))
#                      else: # Sometimes description is directly in the div
#                           specific_details.append(desc_div.get_text(" ", strip=True))


#             elif "flipkart.com" in domain:
#                 # Highlights section
#                 highlights_div = soup.find("div", class_="_2418kt") # Check this class
#                 if highlights_div:
#                     for li in highlights_div.find_all("li", class_="_21Ahn-"): # Check this class
#                         specific_details.append(li.get_text(" ", strip=True))
#                 # Detailed description section
#                 desc_div = soup.find("div", class_="_1AN87F") # Check this class
#                 if desc_div:
#                     paragraphs = desc_div.find_all("p")
#                     if paragraphs:
#                         for p in paragraphs:
#                             specific_details.append(p.get_text(" ", strip=True))
#                     else:
#                          specific_details.append(desc_div.get_text(" ", strip=True))


#             elif "1mg.com" in domain:
#                  # Key Benefits / Product Information sections
#                  for heading_text in ["Product Information", "Key Benefits", "Uses", "Directions for Use", "Safety Information"]:
#                      heading = soup.find(['h2', 'h3', 'strong'], string=re.compile(heading_text, re.I))
#                      if heading:
#                          content_sibling = heading.find_next_sibling(['div', 'ul', 'p'])
#                          if content_sibling:
#                               if content_sibling.name == 'ul':
#                                    for li in content_sibling.find_all('li'):
#                                         specific_details.append(li.get_text(" ", strip=True))
#                               else:
#                                    specific_details.append(content_sibling.get_text(" ", strip=True))

#                  # Fallback description class
#                  desc_div = soup.find("div", class_=re.compile(r"ProductDescription__description")) # Example class, inspect actual page
#                  if desc_div:
#                      specific_details.append(desc_div.get_text(" ", strip=True))

#             elif "netmeds.com" in domain:
#                 # Uses / Benefits sections
#                 for heading_text in ["Uses", "Benefits", "Product Details", "Key Ingredients", "Directions for Use", "Safety Information"]:
#                      heading = soup.find(['h2', 'h3', 'div'], string=re.compile(heading_text, re.I))
#                      if heading:
#                           # Content might be in a sibling div or following p/ul tags
#                           content_parent = heading.find_parent()
#                           next_elems = heading.find_next_siblings(limit=3) # Look nearby
#                           for elem in next_elems:
#                               if elem.name in ['p', 'div', 'ul']:
#                                    if elem.name == 'ul':
#                                         for li in elem.find_all('li'):
#                                              specific_details.append(li.get_text(" ", strip=True))
#                                    else:
#                                         specific_details.append(elem.get_text(" ", strip=True))
#                                    break # Assume first sibling block is the content

#                 # Generic description container
#                 desc_div = soup.find("div", id="product_content") or soup.find("div", class_="inner-content") # Inspect needed
#                 if desc_div:
#                     specific_details.append(desc_div.get_text(" ", strip=True))


#             elif "apollopharmacy.in" in domain:
#                  # Check for description sections like on 1mg/Netmeds
#                  for heading_text in ["Product Details", "Key Benefits", "Directions for Use", "Safety Information", "Key Ingredients"]:
#                      heading = soup.find(['h2', 'h3', 'p', 'strong'], string=re.compile(heading_text, re.I))
#                      if heading:
#                          content_sibling = heading.find_next_sibling(['div', 'ul', 'p'])
#                          if content_sibling:
#                              if content_sibling.name == 'ul':
#                                   for li in content_sibling.find_all('li'):
#                                        specific_details.append(li.get_text(" ", strip=True))
#                              else:
#                                   specific_details.append(content_sibling.get_text(" ", strip=True))

#                  # Fallback class
#                  desc_div = soup.find("div", class_=re.compile("PdpWeb_productInfo__")) # Inspect needed
#                  if desc_div:
#                       specific_details.append(desc_div.get_text(" ", strip=True))


#             elif "pharmeasy.in" in domain:
#                 # Similar structure to 1mg/Apollo usually
#                 for heading_text in ["Product Information", "Key Benefits", "Uses", "Directions for Use", "Safety Information", "Key Ingredients"]:
#                      heading = soup.find(['h2', 'h3', 'div'], string=re.compile(heading_text, re.I))
#                      if heading:
#                           content_sibling = heading.find_next_sibling(['div', 'ul', 'p'])
#                           if content_sibling:
#                               if content_sibling.name == 'ul':
#                                    for li in content_sibling.find_all('li'):
#                                         specific_details.append(li.get_text(" ", strip=True))
#                               else:
#                                    specific_details.append(content_sibling.get_text(" ", strip=True))

#                 desc_div = soup.find("div", class_=re.compile("ProductDescriptionContainer_description")) # Inspect needed
#                 if desc_div:
#                      specific_details.append(desc_div.get_text(" ", strip=True))

#             # Add specific details to the main list
#             details.extend(specific_details)

#         except Exception as e:
#              logging.warning(f"Error during domain-specific detail extraction for {domain}: {e}")


#     # 3. Generic Paragraph Extraction (Low Priority)
#     if len(details) < 3:
#         try:
#             # Look within main content areas first
#             main_content = soup.find("main") or soup.find("article") or soup.find("div", id="dp-container") or soup.find("div", id="content") or soup.body
#             if main_content:
#                 paragraphs = main_content.find_all("p", limit=15) # Limit search space
#                 for p in paragraphs:
#                      text = p.get_text(" ", strip=True)
#                      # Avoid paragraphs that are just links or very short
#                      if text and len(text.split()) > 5 and not p.find('a', recursive=False): # Check if paragraph *itself* is just a link
#                           details.append(text)
#         except Exception as e:
#              logging.warning(f"Error during generic paragraph extraction: {e}")

#     # 4. Refine and Filter
#     refined_details = refine_details(details)


#     # 5. Final Selection
#     # Prioritize health-related keywords if too many details found
#     keywords = ["vitamin", "probiotic", "nutrition", "tablet", "capsule", "supplement", "digest", "health", "blend", "herbal", "ayurvedic", "relief", "support", "mg", "ml", "ingredient", "extract", "mineral", "immunity"]
#     def is_informative(text):
#         return any(kw in text.lower() for kw in keywords)

#     informative_details = [d for d in refined_details if is_informative(d)]

#     # Prefer informative details, but fall back to others if needed
#     if len(informative_details) >= 1:
#         final_details = informative_details
#     else:
#         final_details = refined_details

#     # Limit to max 3-4 details
#     return final_details[:4] if final_details else ["Product details available on the seller's website."]


# def extract_certifications(soup):
#     certs = set()
#     keywords = ["certified", "iso", "fssai", "approved", "authentic", "verified", "legitscript", "gmp", "non-gmo", "organic", "ayush"]
#     try:
#         # Search for text containing keywords
#         for text_node in soup.find_all(string=True):
#             text = text_node.strip()
#             if not text or "{" in text or "}" in text:
#                 continue

#             lower_text = text.lower()
#             if any(kw in lower_text for kw in keywords):
#                 # Try to get a meaningful parent block
#                 parent = text_node.find_parent(['p', 'div', 'span', 'li'])
#                 if parent:
#                     parent_text = parent.get_text(" ", strip=True)
#                     if 15 < len(parent_text) < 150 and any(kw in parent_text.lower() for kw in keywords) and not is_valid_detail(parent_text): # Avoid capturing main details
#                         # Basic cleanup
#                         parent_text = re.sub(r'\s+', ' ', parent_text).strip()
#                         certs.add(parent_text)

#         # Look for images with alt text suggesting certification
#         for img in soup.find_all("img", alt=True):
#              alt_text = img['alt'].lower()
#              if any(kw in alt_text for kw in keywords) and len(img['alt']) < 100:
#                  certs.add(img['alt'].strip())

#     except Exception as e:
#          logging.warning(f"Error extracting certifications: {e}")

#     # Limit results and refine
#     refined_certs = []
#     seen_certs = set()
#     for cert in list(certs)[:5]: # Process a few potential candidates
#         cert_lower = cert.lower()
#         is_duplicate = False
#         for seen in seen_certs:
#             if cert_lower in seen or seen in cert_lower: # Simple substring check for overlap
#                 is_duplicate = True
#                 break
#         if not is_duplicate:
#             refined_certs.append(cert)
#             seen_certs.add(cert_lower)

#     return refined_certs[:2] # Return max 2 distinct certifications

# def extract_price(soup):
#     price = "Price not available"
#     found_price = False

#     # Regex patterns
#     price_pattern_rs = r"(?:₹|Rs\.?)\s*([\d,]+\.?\d*)"
#     price_pattern_only_num = r"^([\d,]+\.?\d*)$" # For matching text that is ONLY the price

#     # 1. High-priority: Meta tags
#     try:
#         meta_selectors = [
#             {"itemprop": "price"},
#             {"property": "product:price:amount"},
#             {"property": "og:price:amount"}
#         ]
#         for sel in meta_selectors:
#             meta = soup.find("meta", attrs=sel)
#             if meta and meta.get("content"):
#                 match = re.search(r"([\d,]+\.?\d*)", meta["content"])
#                 if match:
#                     price_val = match.group(1).replace(',', '').strip()
#                     if float(price_val) > 0: # Basic sanity check
#                          price = f"₹{price_val}"
#                          found_price = True
#                          break
#     except Exception as e:
#         logging.debug(f"Error finding price in meta tags: {e}")

#     if found_price: return price

#     # 2. Common price elements (check multiple attributes and tags)
#     try:
#         selectors = [
#             # Amazon
#             ".a-price .a-offscreen",
#             "span#priceblock_ourprice", "span#priceblock_dealprice", "span#price",
#             # Flipkart
#             "div._30jeq3", "div._16Jk6d",
#             # 1mg / Pharmeasy / Apollo / Netmeds (Inspect specific pages for accuracy)
#             ".Price__price___32PR5", ".style__price-tag___KzOkY", ".PriceDetails__final-price___Q7259",
#             ".final-price", ".price", ".product-price", ".selling-price", ".offer-price", ".best-price",
#             # General classes/ids
#             *[f"[class*='{cls}']" for cls in ['price', 'Price', 'amount', 'Amount']],
#             *[f"[id*='{id_}']" for id_ in ['price', 'Price', 'amount', 'Amount']],
#             # itemprop
#             "[itemprop='price']"
#         ]

#         for sel in selectors:
#              elements = soup.select(sel)
#              for element in elements:
#                  text = element.get_text(strip=True)
#                  if not text: continue

#                  match_rs = re.search(price_pattern_rs, text)
#                  match_num = re.match(price_pattern_only_num, text)

#                  potential_price = None
#                  if match_rs:
#                      potential_price = match_rs.group(1)
#                  elif match_num:
#                      # If only number, check siblings/parents for currency symbol
#                      parent_text = element.find_parent().get_text(" ", strip=True) if element.find_parent() else ""
#                      if "₹" in parent_text or "Rs" in parent_text or "INR" in parent_text:
#                            potential_price = match_num.group(1)

#                  if potential_price:
#                      price_val = potential_price.replace(',', '').strip()
#                      try:
#                          if float(price_val) > 0:
#                              price = f"₹{price_val}"
#                              found_price = True
#                              break # Break from inner loop once found
#                      except ValueError:
#                           continue # Ignore if conversion fails
#              if found_price: break # Break from outer loop

#     except Exception as e:
#         logging.warning(f"Error extracting price from common elements: {e}")


#     # 3. Fallback: Search text nodes
#     if not found_price:
#         try:
#              body_text = soup.body.get_text(" ", strip=True)
#              # Look for price patterns near keywords like "Price:", "MRP:", "Offer Price:"
#              context_matches = re.finditer(r"(?:Price|MRP|Offer Price|Buy for)[:\s]+(?:₹|Rs\.?)\s*([\d,]+\.?\d*)", body_text, re.IGNORECASE)
#              for match in context_matches:
#                  price_val = match.group(1).replace(',', '').strip()
#                  if float(price_val) > 0:
#                      price = f"₹{price_val}"
#                      found_price = True
#                      break
#         except Exception as e:
#             logging.debug(f"Error extracting price from text nodes: {e}")

#     return price

# def extract_rating(soup, domain):
#     rating = "Rating not available"
#     rating_pattern = r"(\d(?:\.\d+)?)" # Matches single digit or digit.digit

#     # 1. High-priority: Meta tag
#     try:
#         meta_rating = soup.find("meta", {"itemprop": "ratingValue"})
#         if meta_rating and meta_rating.get("content"):
#             match = re.search(rating_pattern, meta_rating["content"])
#             if match:
#                 val_str = match.group(1)
#                 try:
#                     val = float(val_str)
#                     if 0 <= val <= 5: # Validate range
#                          # Check for max value if available
#                          meta_max = soup.find("meta", {"itemprop": "bestRating"}) # Or sometimes 'ratingCount'? Inspect schemas
#                          max_val = "/5" # Assume 5 if not specified
#                          if meta_max and meta_max.get("content"):
#                               try: max_val = f"/{int(float(meta_max.get('content')))}"
#                               except: pass # Ignore if max is invalid
#                          rating = f"{val_str}{max_val}"
#                          return rating
#                 except ValueError:
#                      pass # Ignore if not a valid float
#     except Exception as e:
#          logging.debug(f"Error finding rating in meta tags: {e}")

#     # 2. Domain-specific and common elements
#     try:
#         selectors = [
#             # Amazon
#             "#acrPopover .a-icon-alt", # Text like "4.2 out of 5 stars"
#             "span[data-hook='rating-out-of-text']",
#             # Flipkart
#             "div._3LWZlK", # Contains rating like "4.3 ★"
#             # 1mg / Netmeds / Apollo / Pharmeasy (Inspect specific pages)
#             ".Rating__rating-number___G_e6k", # Example class
#             ".CardRating_ratings__", # Example class
#             ".styles__prodRating___", # Example class
#             # General
#             "[itemprop='ratingValue']", # Sometimes on span/div
#             "[class*='rating']", "[class*='Rating']", "[class*='star']", "[class*='Star']"
#         ]

#         for sel in selectors:
#             elements = soup.select(sel)
#             for element in elements:
#                  text = element.get_text(" ", strip=True)
#                  if not text:
#                      # Sometimes rating is in an attribute like 'aria-label'
#                      aria_label = element.get('aria-label', '')
#                      if aria_label: text = aria_label

#                  if text:
#                      match = re.search(rating_pattern, text)
#                      if match:
#                          val_str = match.group(1)
#                          try:
#                              val = float(val_str)
#                              if 0 <= val <= 5: # Validate range (most common scale)
#                                  # Determine scale (usually 5)
#                                  scale = "/5"
#                                  if "out of 10" in text.lower(): scale = "/10"
#                                  # Check if the value itself looks like a rating (e.g. > 1)
#                                  if val >= 1:
#                                      rating = f"{val_str}{scale}"
#                                      return rating # Found valid rating
#                          except ValueError:
#                               continue # Ignore if not float
#     except Exception as e:
#         logging.warning(f"Error extracting rating from common elements: {e}")


#     # 3. Fallback: Search text for "X out of 5" patterns
#     try:
#         text_nodes = soup.find_all(string=re.compile(r"(\d\.?\d+?)\s*(?:out\s+of\s+)?(?:5|five)\s*stars?", re.IGNORECASE))
#         for node in text_nodes:
#              match = re.search(r"(\d\.?\d+?)", node, re.IGNORECASE)
#              if match:
#                   val_str = match.group(1)
#                   try:
#                       val = float(val_str)
#                       if 0 <= val <= 5:
#                           rating = f"{val_str}/5"
#                           return rating
#                   except ValueError:
#                        continue
#     except Exception as e:
#          logging.debug(f"Error extracting rating from 'out of 5' text: {e}")


#     return rating


# def clean_title(title, soup):
#     # 1. Initial cleaning from SERP title
#     cleaned_title = re.sub(r"\s*[-|:•]\s*(?:Buy Online|Online|Store|Shop|Price|Best Price|Reviews|Ratings|Offers|India).*", "", title, flags=re.I)
#     cleaned_title = re.sub(r"\s+\(.*?\)", "", cleaned_title) # Remove content in parentheses if it looks like generic info
#     cleaned_title = re.sub(r"^\s*Amazon\.in[:\s]*", "", cleaned_title, flags=re.I)
#     cleaned_title = re.sub(r"^\s*Flipkart\.com[:\s]*", "", cleaned_title, flags=re.I)
#     cleaned_title = re.sub(r"^\s*1mg[:\s]*", "", cleaned_title, flags=re.I)
#     cleaned_title = re.sub(r"^\s*Netmeds[:\s]*", "", cleaned_title, flags=re.I)
#     cleaned_title = re.sub(r"^\s*Apollo Pharmacy[:\s]*", "", cleaned_title, flags=re.I)
#     cleaned_title = re.sub(r"^\s*Pharmeasy[:\s]*", "", cleaned_title, flags=re.I)

#     # Remove common units/counts if they seem appended, keep if part of name
#     # This is tricky, might remove useful info. Be conservative.
#     # cleaned_title = re.sub(r"\b(\d+)\s*(?:Tablets?|Capsules?|Strips?|ml|mg|gm|Pack\s*of\s*\d+)\b", "", cleaned_title, flags=re.I).strip()

#     cleaned_title = cleaned_title.strip()

#     # 2. Try finding a better title from H1 or specific product title elements
#     product_title_element = None
#     h1_tag = soup.find('h1')
#     # Common title elements (add more as needed based on inspection)
#     selectors = [
#         'h1',
#         '#productTitle', # Amazon
#         '.B_NuCI', # Flipkart (check class)
#         '.ProductTitle__product-title___', # 1mg (check class)
#         '.product-title', '.product-name', '.prodName', # Generic/Other pharmacies
#         '[itemprop="name"]'
#     ]
#     for sel in selectors:
#         element = soup.select_one(sel)
#         if element:
#             potential_title = element.get_text(" ", strip=True)
#             if potential_title and len(potential_title) > 5 and len(potential_title) < 150: # Basic validation
#                 product_title_element = potential_title
#                 break # Take the first good one found

#     # 3. Decide which title to use
#     final_title = cleaned_title # Default to cleaned SERP title
#     if product_title_element:
#         # If H1/product title is significantly different and seems valid, prefer it
#         # Simple heuristic: if product title is longer and not just repeating SERP title start
#         if len(product_title_element) > len(cleaned_title) or not cleaned_title.startswith(product_title_element[:15]):
#             final_title = product_title_element

#     # 4. Final cleanup and length limit
#     final_title = re.sub(r'\s+', ' ', final_title).strip()
#     final_title = final_title[:100] # Limit length

#     # Ensure title is not empty or generic placeholder
#     if not final_title or final_title.lower() in ["product", "details", "search results"]:
#         return "Product Title Unavailable"

#     return final_title


# def product_unavailable(soup):
#     text = soup.get_text(" ", strip=True).lower()
#     keywords = [
#         "currently unavailable", "out of stock", "sold out",
#         "item is unavailable", "notify me when available",
#         "product is not available", "discontinued"
#     ]
#     # Also check for specific elements that might indicate unavailability
#     # Example: soup.select_one("#availability .a-color-price") checks Amazon's red "Currently unavailable" text
#     if soup.select_one("#availability .a-color-price") and "unavailable" in soup.select_one("#availability .a-color-price").get_text().lower():
#         return True
#     # Add similar specific checks for other sites if needed

#     return any(keyword in text for keyword in keywords)


# class MedComparator:
#     def __init__(self):
#         self.session = None
#         try:
#             self.llm = Llama(
#                 model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#                 n_ctx=2048,
#                 n_threads=6,
#                 verbose=False,
#                 temperature=0.6 # Slightly lower temp for more factual analysis
#             )
#         except Exception as e:
#             st.error(f"Failed to load LLM model: {e}. Analysis feature will be unavailable.")
#             self.llm = None
#         self.cache = {}
#         self.headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36", # Updated UA
#             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
#             "Accept-Language": "en-US,en;q=0.9",
#             "Accept-Encoding": "gzip, deflate, br",
#             "Referer": "https://www.google.com/",
#             "DNT": "1", # Do Not Track
#             "Upgrade-Insecure-Requests": "1",
#             "Sec-Fetch-Dest": "document",
#             "Sec-Fetch-Mode": "navigate",
#             "Sec-Fetch-Site": "cross-site", # More realistic headers
#             "Sec-Fetch-User": "?1",
#             "Sec-Ch-Ua": '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"',
#             "Sec-Ch-Ua-Mobile": "?0",
#             "Sec-Ch-Ua-Platform": '"Windows"'
#         }
#         self.request_count = 0
#         self.max_requests_per_domain = 10 # Limit requests per domain per run


#     async def __aenter__(self):
#         connector = aiohttp.TCPConnector(limit_per_host=5) # Limit concurrent connections per host
#         self.session = aiohttp.ClientSession(headers=self.headers, connector=connector)
#         self.domain_request_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
#         return self

#     async def __aexit__(self, *args):
#         await self.session.close()

#     async def search_products(self, query):
#         cache_key = hashlib.sha256(query.encode()).hexdigest()
#         if cache_key in self.cache:
#             logging.info(f"Using cached results for query: {query}")
#             return self.cache[cache_key]

#         logging.info(f"Searching for query: {query}")
#         tasks = []
#         serper_api_key = os.getenv("SERPER_API_KEY")
#         if not serper_api_key:
#             st.error("SERPER_API_KEY not found in environment variables.")
#             return []

#         for domain in ALLOWED_DOMAINS:
#             params = {
#                 "q": f"{query} site:{domain}",
#                 "num": 5, # Request fewer results per domain initially
#                 "hl": "en",
#                 "gl": "in",
#                 "api_key": serper_api_key
#             }
#             tasks.append(self.fetch_domain_results(params))

#         results_nested = await asyncio.gather(*tasks)
#         all_products_raw = [p for sublist in results_nested for p in sublist]

#         # Process and filter results
#         processed_products = await self.process_serp_results(all_products_raw)
#         filtered_products = self.filter_products(processed_products)

#         self.cache[cache_key] = filtered_products
#         logging.info(f"Found {len(filtered_products)} products after filtering for query: {query}")
#         return filtered_products

#     def filter_products(self, products):
#         filtered = []
#         seen_links = set()
#         seen_titles_sources = set()
#         domain_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
#         max_per_source = 3 # Max products from the same source (e.g., 3 from Amazon)
#         total_max = 10 # Max total products

#         for p in products:
#             if not p or not p.get("link") or not p.get("title") or p["title"] == "Product Title Unavailable":
#                 continue

#             link = p["link"]
#             source_domain = extract_domain(link)
#             title_source_key = f"{p['title'].lower()}-{p['source'].lower()}"

#             # Basic Deduplication and Limits
#             if link in seen_links:
#                 continue
#             if title_source_key in seen_titles_sources:
#                 continue
#             if domain_counts.get(source_domain, 0) >= max_per_source:
#                 continue
#             if len(filtered) >= total_max:
#                 break

#             # Additional Quality Checks (Optional)
#             if p['price'] == "Price not available" and p['rating'] == "Rating not available" and len(p['details']) <= 1:
#                  logging.info(f"Skipping low-information product: {p['title']}")
#                  continue # Skip if very little info was scraped

#             filtered.append(p)
#             seen_links.add(link)
#             seen_titles_sources.add(title_source_key)
#             if source_domain in domain_counts:
#                  domain_counts[source_domain] += 1

#         # Sort by price or relevance if desired (optional)
#         # filtered.sort(key=lambda x: float(x['price'].replace('₹','').replace(',','')) if x['price'] != 'Price not available' else float('inf'))

#         return filtered

#     async def fetch_domain_results(self, params):
#         domain = params['q'].split("site:")[-1]
#         logging.info(f"Fetching SERP for {domain}...")
#         try:
#             async with self.session.post("https://google.serper.dev/search", json=params) as res: # Use POST as recommended
#                 if res.status == 200:
#                     results = await res.json()
#                     logging.info(f"Got {len(results.get('organic', []))} results from SERP for {domain}")
#                     return results.get("organic", [])
#                 else:
#                     logging.error(f"SERP API request failed for {domain} with status {res.status}: {await res.text()}")
#                     return []
#         except aiohttp.ClientError as e:
#             logging.error(f"Network error during SERP API request for {domain}: {e}")
#             return []
#         except Exception as e:
#             logging.error(f"Unexpected error during SERP API request for {domain}: {e}")
#             return []

#     async def process_serp_results(self, organic_results):
#         tasks = []
#         valid_links = set()
#         for result in organic_results:
#             link = result.get("link")
#             title = result.get("title", "")
#             if not link or not title:
#                  continue

#             if is_allowed_url(link) and not is_generic_result(title, link) and link not in valid_links:
#                  source_domain = extract_domain(link)
#                  if self.domain_request_counts.get(source_domain, 0) < self.max_requests_per_domain:
#                       self.domain_request_counts[source_domain] = self.domain_request_counts.get(source_domain, 0) + 1
#                       tasks.append(self.scrape_product_page(result))
#                       valid_links.add(link)
#                  else:
#                       logging.warning(f"Skipping scrape for {link} due to domain request limit.")


#         scraped_products = await asyncio.gather(*tasks)
#         return [p for p in scraped_products if p] # Filter out None results


#     async def scrape_product_page(self, result):
#         link = result["link"]
#         serp_title = result["title"]
#         logging.info(f"Scraping: {link}")
#         try:
#             # Add slight delay to be polite
#             await asyncio.sleep(0.5 + self.request_count * 0.1)
#             self.request_count += 1

#             async with self.session.get(link, timeout=25, allow_redirects=True) as res: # Reduced timeout slightly
#                 if res.status != 200:
#                     logging.warning(f"Failed to fetch {link} - Status: {res.status}")
#                     return None
#                 try:
#                     # Read with appropriate encoding (try common ones)
#                     html_bytes = await res.read()
#                     try:
#                         html = html_bytes.decode('utf-8')
#                     except UnicodeDecodeError:
#                          try:
#                               html = html_bytes.decode('iso-8859-1')
#                          except UnicodeDecodeError:
#                               html = html_bytes.decode('cp1252', errors='ignore') # Fallback

#                 except Exception as e:
#                     logging.error(f"Failed to read/decode HTML from {link}: {e}")
#                     return None

#                 if not html:
#                     logging.warning(f"Empty HTML content received from {link}")
#                     return None

#                 soup = BeautifulSoup(html, "lxml")

#                 if product_unavailable(soup):
#                     logging.info(f"Product unavailable: {link}")
#                     return None

#                 domain = extract_domain(link)
#                 title = clean_title(serp_title, soup)

#                 if title == "Product Title Unavailable":
#                      logging.warning(f"Could not extract a valid title for {link}")
#                      return None # Skip if no usable title

#                 price = extract_price(soup)
#                 rating = extract_rating(soup, domain)
#                 details = extract_product_details(soup, domain)
#                 certifications = extract_certifications(soup)

#                 # Final check: ensure essential info exists
#                 if price == "Price not available" and rating == "Rating not available" and len(details) <= 1:
#                       logging.info(f"Skipping product due to lack of essential info: {title} ({link})")
#                       return None

#                 product_data = {
#                     "title": title,
#                     "link": link,
#                     "price": price,
#                     "rating": rating,
#                     "details": details, # Renamed from 'features'
#                     "certifications": certifications,
#                     "source": domain.split('.')[0].title() if '.' in domain else domain.title() # More robust source name
#                 }
#                 logging.info(f"Successfully scraped: {title} ({product_data['price']}, {product_data['rating']})")
#                 return product_data

#         except asyncio.TimeoutError:
#             logging.warning(f"Timeout scraping {link}")
#             return None
#         except aiohttp.ClientError as e:
#             logging.warning(f"Client error scraping {link}: {e}")
#             return None
#         except Exception as e:
#             logging.error(f"Unexpected error scraping {link}: {e}", exc_info=True) # Log traceback for unexpected errors
#             return None
#         finally:
#             self.request_count -= 1 # Decrement counter after request finishes or fails


# async def main():
#     st.set_page_config(page_title="MedCompare Pro", page_icon="💊", layout="wide")
#     st.title("🌍 MedCompare - Health Supplement Analyzer")

#     # Initialize comparator in session state to persist across reruns
#     if 'comparator' not in st.session_state:
#         with st.spinner("Initializing medical comparison engine..."):
#             st.session_state.comparator = MedComparator()

#     comparator = st.session_state.comparator

#     # Use a form to prevent reruns on every widget interaction
#     with st.form(key="search_form"):
#         query = st.text_input(
#             "Enter medicine/supplement name or health concern:",
#             key="query_input",
#             placeholder="E.g.: 'Vitamin C 500mg tablets', 'Probiotics for women', 'Piles relief ayurvedic'"
#         )
#         submit_button = st.form_submit_button("Search Products")

#     if submit_button and query:
#         st.session_state.products = None # Reset products on new search
#         st.session_state.analysis = None # Reset analysis
#         st.session_state.current_query = query # Store the query that triggered the search

#         async with comparator: # Use async context manager
#             with st.spinner(f"Searching for '{query}'... This may take a minute."):
#                 adjusted_query = adjust_query(query)
#                 logging.info(f"Adjusted query: {adjusted_query}")
#                 products = await comparator.search_products(adjusted_query)
#                 st.session_state.products = products

#     # Display results if products exist in session state for the current query
#     if 'products' in st.session_state and st.session_state.products is not None:
#         products = st.session_state.products
#         if not products:
#             st.error(f"No specific, available products found for '{st.session_state.current_query}'. Try:")
#             st.markdown("""
#                 * Be more specific (e.g., include 'tablets', 'syrup', dosage '500mg', quantity '60 units').
#                 * Try adding a known brand name (e.g., 'Himalaya Pilex', 'Carbamide Forte').
#                 * Broaden the search slightly if too specific (e.g., 'best multivitamin' instead of a very niche one).
#                 * Check your spelling or try different phrasing.
#             """)
#         else:
#             st.markdown(f"Found **{len(products)}** relevant health products for '{st.session_state.current_query}':")
#             st.markdown("---")

#             for i, product in enumerate(products):
#                 with st.container():
#                     st.subheader(f"{i+1}. {product['title']}")
#                     col1, col2, col3 = st.columns([2, 1, 1])
#                     col1.markdown(f"**Source:** {product['source']}")
#                     col2.markdown(f"**Price:** {product['price']}")
#                     col3.markdown(f"**Rating:** {product['rating']}")

#                     expander = st.expander("Show Details", expanded=False)
#                     with expander:
#                          expander.markdown(f"**Details:**")
#                          if product['details']:
#                              for detail in product['details']:
#                                  expander.markdown(f"- {detail}")
#                          else:
#                              expander.markdown("- No specific details extracted.")

#                          if product["certifications"]:
#                              expander.markdown("**Certifications:**")
#                              for cert in product["certifications"]:
#                                  expander.markdown(f"- 🛡️ {cert}")
#                          else:
#                               expander.markdown("**Certifications:** - None found.")

#                          expander.markdown(f"🔗 [View Product Page]({product['link']})", unsafe_allow_html=True)
#                     st.markdown("---")


#              # Prepare data for the full comparison table (displayed before analysis)
#             if products: # Check if products list is not empty
#                  display_data = [{
#                      "Title": p["title"],
#                      "Price": p["price"],
#                      "Rating": p["rating"],
#                      "Details": " | ".join(p["details"][:2]), # Show only first 2 details in table
#                      "Certifications": ", ".join(p["certifications"]),
#                      "Source": p["source"],
#                      "Link": p["link"] # Include link for potential clicking/reference
#                  } for p in products] # USE ALL PRODUCTS FOR DISPLAY DF

#                  df_display = pd.DataFrame(display_data)

#                  st.subheader("Full Product Comparison Table")
#                  # Display the full table - consider making columns clickable or formatting price/rating
#                  st.dataframe(df_display, use_container_width=True, column_config={
#                      "Link": st.column_config.LinkColumn("View Product", display_text="🔗")
#                  })
#                  st.markdown("---")


#             # Analysis Section (only if products found and LLM loaded)
#             if products and comparator.llm:
#                  button_key = "generate_analysis_full_v4" # New key
#                  generate_analysis_button = st.button("💡 Generate Recommendation (Fast & Concise)", key=button_key) # Changed button text

#                  session_state_key = 'analysis_full_v4' # Corresponding session state key

#                  if (session_state_key not in st.session_state or st.session_state[session_state_key] is None) and generate_analysis_button:
#                      # Prepare data for LLM - Send core info ONLY
#                      llm_analysis_data = [{
#                          "Title": p["title"],
#                          "Price": p["price"],
#                          "Rating": p["rating"],
#                          "Source": p.get("source", "N/A") # Use .get for safety
#                          # DO NOT send Key_Feature to simplify the task for speed
#                      } for p in products]

#                      if not llm_analysis_data:
#                          st.warning("Not enough product data to generate analysis.")
#                      else:
#                          with st.spinner("💡 Generating concise recommendation (Aiming for < 1 min)..."):
#                              # Create prompt data string (simpler now)
#                              prompt_data = "\n".join([
#                                  f"- {p['Title']} (Source: {p['Source']}, Price: {p['Price']}, Rating: {p['Rating']})"
#                                  for p in llm_analysis_data
#                              ])

#                              if not prompt_data:
#                                  st.error("Could not prepare data for analysis.")
#                              else:
#                                  # *** REVISED PROMPT - Simplified Justification Requirement ***
#                                  user_query = st.session_state.get('current_query', 'health products')
#                                  prompt = f"""You are an efficient health product analyst providing a quick recommendation for a user who searched for '{user_query}'.
# Review these products based ONLY on Title, Source, Price, and Rating:

# {prompt_data}

# Instructions:
# 1. **Directly recommend 1-2 specific products** from the list that seem most suitable for '{user_query}'.
# 2. **Briefly justify your recommendation (1 sentence per product max)** using Price, Rating, or apparent relevance from the product Title/Source. (Do NOT analyze features you don't have).
# 3. Keep the *entire response* extremely brief and focused only on the recommendation and its justification (target 2-3 sentences total).
# 4. Be direct and objective. Do not echo instructions.

# Concise Recommendation:
# """
#                                  try:
#                                      # *** Adjusted max_tokens - may need tuning ***
#                                      response = comparator.llm(
#                                          prompt,
#                                          max_tokens=200, # Reduced slightly as task is simpler
#                                          stop=["\n\n", "Instructions:", "---", "IMPORTANT:"],
#                                          temperature=0.6
#                                      )
#                                      analysis_text = response['choices'][0]['text'].strip()

#                                      # Basic cleanup
#                                      # ... (cleanup logic as before) ...
#                                      if analysis_text and not analysis_text.endswith(('.', '!', '?','.',':')):
#                                           last_sentence_end = max(analysis_text.rfind('.'), analysis_text.rfind('?'), analysis_text.rfind('!'))
#                                           if last_sentence_end > 0:
#                                                analysis_text = analysis_text[:last_sentence_end+1]
#                                           else:
#                                                analysis_text += '...'


#                                      # Add the disclaimer
#                                      analysis_text += "\n\n**IMPORTANT:** This is an AI-generated recommendation based on limited scraped data (Price, Rating, Title) and NOT medical advice. Always consult a healthcare professional or pharmacist for personalized guidance."
#                                      st.session_state[session_state_key] = analysis_text
#                                      st.rerun()
#                                  except Exception as e:
#                                      logging.error(f"LLM analysis failed: {e}")
#                                      st.error(f"Could not generate AI analysis: {e}")
#                                      st.session_state[session_state_key] = "Analysis generation failed."

#                  # Display the analysis
#                  if session_state_key in st.session_state and st.session_state[session_state_key]:
#                      st.subheader("AI Concise Recommendation") # Simpler title
#                      st.markdown(st.session_state[session_state_key])
#                      st.markdown("---")

# if __name__ == "__main__":
#     # Setup basic logging to see progress/errors in console
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     # Set higher level for noisy libraries
#     logging.getLogger("urllib3").setLevel(logging.WARNING)
#     logging.getLogger("asyncio").setLevel(logging.WARNING)
#     logging.getLogger("aiohttp.client").setLevel(logging.WARNING)

#     # Check for API key existence early
#     if not os.getenv("SERPER_API_KEY"):
#         st.error("Fatal Error: SERPER_API_KEY environment variable is not set. The application cannot search for products.")
#         st.stop()

#     # Run the async main function
#     try:
#         asyncio.run(main())
#     except Exception as e:
#         st.error(f"An unexpected error occurred: {e}")
#         logging.error("Application crashed", exc_info=True)









import os
import re
import json
import aiohttp
import asyncio
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import hashlib
import nest_asyncio
from urllib.parse import urlparse
from textblob import TextBlob
import logging
from groq import Groq

nest_asyncio.apply()
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_DOMAINS = [
    "amazon.in", "flipkart.com", "1mg.com",
    "netmeds.com", "apollopharmacy.in", "pharmeasy.in",
    "medplusmart.com", "walmart.com"
]

def correct_spelling(text):
    try:
        return str(TextBlob(text).correct())
    except Exception as e:
        logging.warning(f"Spelling correction failed: {e}")
        return text

def adjust_query(user_query):
    corrected = correct_spelling(user_query)
    base = re.sub(r"\b(compare|vs|versus|suggest me \d+|show me|which is better|that are there|find|get me)\b", "", corrected, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    if not any(domain in base for domain in ["amazon", "flipkart", "1mg", "netmeds", "apollo", "pharmeasy", "medplus"]):
         base = f"{base} buy online india"
    return base.strip()

def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '').replace('m.', '').split(':')[0]
        if domain == "apollo.pharmacy":
             return "apollopharmacy.in"
        return domain
    except Exception:
        return ""

def is_allowed_url(url):
    domain = extract_domain(url)
    return any(allowed == domain for allowed in ALLOWED_DOMAINS)

def is_generic_result(title, url):
    generic_terms = [
        "bestsellers", "product price list", "shop online", "trusted online",
        "order medicine", "category", "all products", "health conditions",
        "store near you", "pharmacy online", "medicine list"
    ]
    combined = (title + " " + url).lower()
    if title.lower() in ["medicines", "health products", "online pharmacy", "supplements"]:
        return True
    if any(f"/{term}/" in url.lower() for term in ["category", "categories", "all", "list"]):
         return True
    return any(term in combined for term in generic_terms)

def is_valid_detail(sentence):
    sentence = sentence.strip()
    if len(sentence) < 25 or len(sentence) > 300:
        return False

    stop_phrases = [
        "privacy policy", "terms of use", "return policy", "add to cart", "in stock",
        "buy now", "tell us about", "lower price", "csrf", "added to cart", "shipping information",
        "customer reviews", "related products", "login", "register", "track order", "specifications",
        "highlights", "limited time offer", "best price guaranteed", "shop now", "free delivery",
        "secure transaction", "go back", "view details", "learn more", "click here",
        "frequently asked questions", "faq", "contact us", "about us", "need help",
        "powered by", "copyright", "all rights reserved", "sign in", "your account",
        "compare prices", "check availability", "select options", "quantity",
        "description", "features", "key benefits",
        "how to use", "safety information", "ingredients", "directions for use",
        "subscribe", "newsletter", "download app", "visit store"
    ]

    lower_sentence = sentence.lower()
    if lower_sentence in stop_phrases:
        return False
    if any(phrase in lower_sentence for phrase in stop_phrases):
        return False

    if re.match(r"^\W*(\*|\-|•|>|\d+\.)\s+", sentence):
        return False
    if sentence.count('{') > 0 or sentence.count('}') > 0 or sentence.count('[') > 1 or sentence.count(']') > 1:
        return False
    if sentence.count('/') > 3 or sentence.count(':') > 3:
        return False

    if re.search(r"(?:best|top|great|amazing|fantastic)\s+(?:deals?|offers?|price|quality)", lower_sentence):
        return False
    if re.search(r"shop|buy|order\s+now", lower_sentence):
        return False
    if "click" in lower_sentence and "here" in lower_sentence:
        return False

    return True

def refine_details(details):
    refined = []
    for d in details:
        d = d.replace("View more", "").replace("Read more", "").strip()
        d = re.sub(r'\s+', ' ', d)
        if d and is_valid_detail(d):
             refined.append(d)

    seen = set()
    final = []
    for d in refined:
        simplified_d = ''.join(filter(str.isalnum, d.lower()))
        if simplified_d and simplified_d not in seen:
            seen.add(simplified_d)
            final.append(d)

    final.sort(key=len, reverse=True)
    return final

def extract_product_details(soup, domain):
    details = []

    try:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    product_data = None
                    queue = data[:]
                    while queue:
                        item = queue.pop(0)
                        if isinstance(item, dict):
                            if item.get("@type") == "Product":
                                product_data = item
                                break
                            if "@graph" in item and isinstance(item["@graph"], list):
                                queue.extend(item["@graph"])
                        elif isinstance(item, list):
                            queue.extend(item)

                elif isinstance(data, dict) and data.get("@type") == "Product":
                     product_data = data
                else:
                    product_data = None

                if product_data and (desc := product_data.get("description")):
                    sentences = re.split(r'(?<=[.!?])\s+', desc)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:
                            details.append(sentence)
                    if details: break

            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                 logging.debug(f"LD+JSON parsing error: {e}")
                 continue
    except Exception as e:
        logging.error(f"Error processing script tags: {e}")

    if len(details) < 3:
        specific_details = []
        try:
            if "amazon.in" in domain:
                bullets_div = soup.find("div", id="feature-bullets")
                if bullets_div:
                    for li in bullets_div.find_all("li"):
                        text = li.get_text(" ", strip=True)
                        if text and not text.lower().startswith("see more product details"):
                            specific_details.append(text)
                desc_div = soup.find("div", id="productDescription")
                if desc_div:
                     paragraphs = desc_div.find_all("p")
                     if paragraphs:
                          for p in paragraphs:
                               specific_details.append(p.get_text(" ", strip=True))
                     else:
                          specific_details.append(desc_div.get_text(" ", strip=True))

            elif "flipkart.com" in domain:
                highlights_div = soup.find("div", class_="_2418kt")
                if highlights_div:
                    for li in highlights_div.find_all("li", class_="_21Ahn-"):
                        specific_details.append(li.get_text(" ", strip=True))
                desc_div = soup.find("div", class_="_1AN87F")
                if desc_div:
                    paragraphs = desc_div.find_all("p")
                    if paragraphs:
                        for p in paragraphs:
                            specific_details.append(p.get_text(" ", strip=True))
                    else:
                         specific_details.append(desc_div.get_text(" ", strip=True))

            elif "1mg.com" in domain:
                 for heading_text in ["Product Information", "Key Benefits", "Uses", "Directions for Use", "Safety Information"]:
                     heading = soup.find(['h2', 'h3', 'strong'], string=re.compile(heading_text, re.I))
                     if heading:
                         content_sibling = heading.find_next_sibling(['div', 'ul', 'p'])
                         if content_sibling:
                              if content_sibling.name == 'ul':
                                   for li in content_sibling.find_all('li'):
                                        specific_details.append(li.get_text(" ", strip=True))
                              else:
                                   specific_details.append(content_sibling.get_text(" ", strip=True))

                 desc_div = soup.find("div", class_=re.compile(r"ProductDescription__description"))
                 if desc_div:
                     specific_details.append(desc_div.get_text(" ", strip=True))

            elif "netmeds.com" in domain:
                for heading_text in ["Uses", "Benefits", "Product Details", "Key Ingredients", "Directions for Use", "Safety Information"]:
                     heading = soup.find(['h2', 'h3', 'div'], string=re.compile(heading_text, re.I))
                     if heading:
                          content_parent = heading.find_parent()
                          next_elems = heading.find_next_siblings(limit=3)
                          for elem in next_elems:
                              if elem.name in ['p', 'div', 'ul']:
                                   if elem.name == 'ul':
                                        for li in elem.find_all('li'):
                                             specific_details.append(li.get_text(" ", strip=True))
                                   else:
                                        specific_details.append(elem.get_text(" ", strip=True))
                                   break

                desc_div = soup.find("div", id="product_content") or soup.find("div", class_="inner-content")
                if desc_div:
                    specific_details.append(desc_div.get_text(" ", strip=True))

            elif "apollopharmacy.in" in domain:
                 for heading_text in ["Product Details", "Key Benefits", "Directions for Use", "Safety Information", "Key Ingredients"]:
                     heading = soup.find(['h2', 'h3', 'p', 'strong'], string=re.compile(heading_text, re.I))
                     if heading:
                         content_sibling = heading.find_next_sibling(['div', 'ul', 'p'])
                         if content_sibling:
                             if content_sibling.name == 'ul':
                                  for li in content_sibling.find_all('li'):
                                       specific_details.append(li.get_text(" ", strip=True))
                             else:
                                  specific_details.append(content_sibling.get_text(" ", strip=True))

                 desc_div = soup.find("div", class_=re.compile("PdpWeb_productInfo__"))
                 if desc_div:
                      specific_details.append(desc_div.get_text(" ", strip=True))

            elif "pharmeasy.in" in domain:
                for heading_text in ["Product Information", "Key Benefits", "Uses", "Directions for Use", "Safety Information", "Key Ingredients"]:
                     heading = soup.find(['h2', 'h3', 'div'], string=re.compile(heading_text, re.I))
                     if heading:
                          content_sibling = heading.find_next_sibling(['div', 'ul', 'p'])
                          if content_sibling:
                              if content_sibling.name == 'ul':
                                   for li in content_sibling.find_all('li'):
                                        specific_details.append(li.get_text(" ", strip=True))
                              else:
                                   specific_details.append(content_sibling.get_text(" ", strip=True))

                desc_div = soup.find("div", class_=re.compile("ProductDescriptionContainer_description"))
                if desc_div:
                     specific_details.append(desc_div.get_text(" ", strip=True))

            details.extend(specific_details)

        except Exception as e:
             logging.warning(f"Error during domain-specific detail extraction for {domain}: {e}")

    if len(details) < 3:
        try:
            main_content = soup.find("main") or soup.find("article") or soup.find("div", id="dp-container") or soup.find("div", id="content") or soup.body
            if main_content:
                paragraphs = main_content.find_all("p", limit=15)
                for p in paragraphs:
                     text = p.get_text(" ", strip=True)
                     if text and len(text.split()) > 5 and not p.find('a', recursive=False):
                          details.append(text)
        except Exception as e:
             logging.warning(f"Error during generic paragraph extraction: {e}")

    refined_details = refine_details(details)

    keywords = ["vitamin", "probiotic", "nutrition", "tablet", "capsule", "supplement", "digest", "health", "blend", "herbal", "ayurvedic", "relief", "support", "mg", "ml", "ingredient", "extract", "mineral", "immunity"]
    def is_informative(text):
        return any(kw in text.lower() for kw in keywords)

    informative_details = [d for d in refined_details if is_informative(d)]

    if len(informative_details) >= 1:
        final_details = informative_details
    else:
        final_details = refined_details

    return final_details[:4] if final_details else ["Product details available on the seller's website."]

def extract_certifications(soup):
    certs = set()
    keywords = ["certified", "iso", "fssai", "approved", "authentic", "verified", "legitscript", "gmp", "non-gmo", "organic", "ayush"]
    try:
        for text_node in soup.find_all(string=True):
            text = text_node.strip()
            if not text or "{" in text or "}" in text:
                continue

            lower_text = text.lower()
            if any(kw in lower_text for kw in keywords):
                parent = text_node.find_parent(['p', 'div', 'span', 'li'])
                if parent:
                    parent_text = parent.get_text(" ", strip=True)
                    if 15 < len(parent_text) < 150 and any(kw in parent_text.lower() for kw in keywords) and not is_valid_detail(parent_text):
                        parent_text = re.sub(r'\s+', ' ', parent_text).strip()
                        certs.add(parent_text)

        for img in soup.find_all("img", alt=True):
             alt_text = img['alt'].lower()
             if any(kw in alt_text for kw in keywords) and len(img['alt']) < 100:
                 certs.add(img['alt'].strip())

    except Exception as e:
         logging.warning(f"Error extracting certifications: {e}")

    refined_certs = []
    seen_certs = set()
    for cert in list(certs)[:5]:
        cert_lower = cert.lower()
        is_duplicate = False
        for seen in seen_certs:
            if cert_lower in seen or seen in cert_lower:
                is_duplicate = True
                break
        if not is_duplicate:
            refined_certs.append(cert)
            seen_certs.add(cert_lower)

    return refined_certs[:2]

def extract_price(soup):
    price = "Price not available"
    found_price = False

    price_pattern_rs = r"(?:₹|Rs\.?)\s*([\d,]+\.?\d*)"
    price_pattern_only_num = r"^([\d,]+\.?\d*)$"

    try:
        meta_selectors = [
            {"itemprop": "price"},
            {"property": "product:price:amount"},
            {"property": "og:price:amount"}
        ]
        for sel in meta_selectors:
            meta = soup.find("meta", attrs=sel)
            if meta and meta.get("content"):
                match = re.search(r"([\d,]+\.?\d*)", meta["content"])
                if match:
                    price_val = match.group(1).replace(',', '').strip()
                    if float(price_val) > 0:
                         price = f"₹{price_val}"
                         found_price = True
                         break
    except Exception as e:
        logging.debug(f"Error finding price in meta tags: {e}")

    if found_price: return price

    try:
        selectors = [
            ".a-price .a-offscreen",
            "span#priceblock_ourprice", "span#priceblock_dealprice", "span#price",
            "div._30jeq3", "div._16Jk6d",
            ".Price__price___32PR5", ".style__price-tag___KzOkY", ".PriceDetails__final-price___Q7259",
            ".final-price", ".price", ".product-price", ".selling-price", ".offer-price", ".best-price",
            *[f"[class*='{cls}']" for cls in ['price', 'Price', 'amount', 'Amount']],
            *[f"[id*='{id_}']" for id_ in ['price', 'Price', 'amount', 'Amount']],
            "[itemprop='price']"
        ]

        for sel in selectors:
             elements = soup.select(sel)
             for element in elements:
                 text = element.get_text(strip=True)
                 if not text: continue

                 match_rs = re.search(price_pattern_rs, text)
                 match_num = re.match(price_pattern_only_num, text)

                 potential_price = None
                 if match_rs:
                     potential_price = match_rs.group(1)
                 elif match_num:
                     parent_text = element.find_parent().get_text(" ", strip=True) if element.find_parent() else ""
                     if "₹" in parent_text or "Rs" in parent_text or "INR" in parent_text:
                           potential_price = match_num.group(1)

                 if potential_price:
                     price_val = potential_price.replace(',', '').strip()
                     try:
                         if float(price_val) > 0:
                             price = f"₹{price_val}"
                             found_price = True
                             break
                     except ValueError:
                          continue
             if found_price: break

    except Exception as e:
        logging.warning(f"Error extracting price from common elements: {e}")

    if not found_price:
        try:
             body_text = soup.body.get_text(" ", strip=True)
             context_matches = re.finditer(r"(?:Price|MRP|Offer Price|Buy for)[:\s]+(?:₹|Rs\.?)\s*([\d,]+\.?\d*)", body_text, re.IGNORECASE)
             for match in context_matches:
                 price_val = match.group(1).replace(',', '').strip()
                 if float(price_val) > 0:
                     price = f"₹{price_val}"
                     found_price = True
                     break
        except Exception as e:
            logging.debug(f"Error extracting price from text nodes: {e}")

    return price

def extract_rating(soup, domain):
    rating = "Rating not available"
    rating_pattern = r"(\d(?:\.\d+)?)"

    try:
        meta_rating = soup.find("meta", {"itemprop": "ratingValue"})
        if meta_rating and meta_rating.get("content"):
            match = re.search(rating_pattern, meta_rating["content"])
            if match:
                val_str = match.group(1)
                try:
                    val = float(val_str)
                    if 0 <= val <= 5:
                         meta_max = soup.find("meta", {"itemprop": "bestRating"})
                         max_val = "/5"
                         if meta_max and meta_max.get("content"):
                              try: max_val = f"/{int(float(meta_max.get('content')))}"
                              except: pass
                         rating = f"{val_str}{max_val}"
                         return rating
                except ValueError:
                     pass
    except Exception as e:
         logging.debug(f"Error finding rating in meta tags: {e}")

    try:
        selectors = [
            "#acrPopover .a-icon-alt",
            "span[data-hook='rating-out-of-text']",
            "div._3LWZlK",
            ".Rating__rating-number___G_e6k",
            ".CardRating_ratings__",
            ".styles__prodRating___",
            "[itemprop='ratingValue']",
            "[class*='rating']", "[class*='Rating']", "[class*='star']", "[class*='Star']"
        ]

        for sel in selectors:
            elements = soup.select(sel)
            for element in elements:
                 text = element.get_text(" ", strip=True)
                 if not text:
                     aria_label = element.get('aria-label', '')
                     if aria_label: text = aria_label

                 if text:
                     match = re.search(rating_pattern, text)
                     if match:
                         val_str = match.group(1)
                         try:
                             val = float(val_str)
                             if 0 <= val <= 5:
                                 scale = "/5"
                                 if "out of 10" in text.lower(): scale = "/10"
                                 if val >= 1:
                                     rating = f"{val_str}{scale}"
                                     return rating
                         except ValueError:
                              continue
    except Exception as e:
        logging.warning(f"Error extracting rating from common elements: {e}")

    try:
        text_nodes = soup.find_all(string=re.compile(r"(\d\.?\d+?)\s*(?:out\s+of\s+)?(?:5|five)\s*stars?", re.IGNORECASE))
        for node in text_nodes:
             match = re.search(r"(\d\.?\d+?)", node, re.IGNORECASE)
             if match:
                  val_str = match.group(1)
                  try:
                      val = float(val_str)
                      if 0 <= val <= 5:
                          rating = f"{val_str}/5"
                          return rating
                  except ValueError:
                       continue
    except Exception as e:
         logging.debug(f"Error extracting rating from 'out of 5' text: {e}")

    return rating

def clean_title(title, soup):
    cleaned_title = re.sub(r"\s*[-|:•]\s*(?:Buy Online|Online|Store|Shop|Price|Best Price|Reviews|Ratings|Offers|India).*", "", title, flags=re.I)
    cleaned_title = re.sub(r"\s+\(.*?\)", "", cleaned_title)
    cleaned_title = re.sub(r"^\s*Amazon\.in[:\s]*", "", cleaned_title, flags=re.I)
    cleaned_title = re.sub(r"^\s*Flipkart\.com[:\s]*", "", cleaned_title, flags=re.I)
    cleaned_title = re.sub(r"^\s*1mg[:\s]*", "", cleaned_title, flags=re.I)
    cleaned_title = re.sub(r"^\s*Netmeds[:\s]*", "", cleaned_title, flags=re.I)
    cleaned_title = re.sub(r"^\s*Apollo Pharmacy[:\s]*", "", cleaned_title, flags=re.I)
    cleaned_title = re.sub(r"^\s*Pharmeasy[:\s]*", "", cleaned_title, flags=re.I)

    cleaned_title = cleaned_title.strip()

    product_title_element = None
    h1_tag = soup.find('h1')
    selectors = [
        'h1',
        '#productTitle',
        '.B_NuCI',
        '.ProductTitle__product-title___',
        '.product-title', '.product-name', '.prodName',
        '[itemprop="name"]'
    ]
    for sel in selectors:
        element = soup.select_one(sel)
        if element:
            potential_title = element.get_text(" ", strip=True)
            if potential_title and len(potential_title) > 5 and len(potential_title) < 150:
                product_title_element = potential_title
                break

    final_title = cleaned_title
    if product_title_element:
        if len(product_title_element) > len(cleaned_title) or not cleaned_title.startswith(product_title_element[:15]):
            final_title = product_title_element

    final_title = re.sub(r'\s+', ' ', final_title).strip()
    final_title = final_title[:100]

    if not final_title or final_title.lower() in ["product", "details", "search results"]:
        return "Product Title Unavailable"

    return final_title

def product_unavailable(soup):
    text = soup.get_text(" ", strip=True).lower()
    keywords = [
        "currently unavailable", "out of stock", "sold out",
        "item is unavailable", "notify me when available",
        "product is not available", "discontinued"
    ]
    if soup.select_one("#availability .a-color-price") and "unavailable" in soup.select_one("#availability .a-color-price").get_text().lower():
        return True
    return any(keyword in text for keyword in keywords)

class MedComparator:
    def __init__(self):
        self.session = None
        self.groq_client = None
        try:
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}. Analysis feature will be unavailable.")
        self.cache = {}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '" Not A;Brand";v="99", "Chromium";v="100", "Google Chrome";v="100"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"'
        }
        self.request_count = 0
        self.max_requests_per_domain = 10

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit_per_host=5)
        self.session = aiohttp.ClientSession(headers=self.headers, connector=connector)
        self.domain_request_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def search_products(self, query):
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        if cache_key in self.cache:
            logging.info(f"Using cached results for query: {query}")
            return self.cache[cache_key]

        logging.info(f"Searching for query: {query}")
        tasks = []
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            st.error("SERPER_API_KEY not found in environment variables.")
            return []

        for domain in ALLOWED_DOMAINS:
            params = {
                "q": f"{query} site:{domain}",
                "num": 5,
                "hl": "en",
                "gl": "in",
                "api_key": serper_api_key
            }
            tasks.append(self.fetch_domain_results(params))

        results_nested = await asyncio.gather(*tasks)
        all_products_raw = [p for sublist in results_nested for p in sublist]

        processed_products = await self.process_serp_results(all_products_raw)
        filtered_products = self.filter_products(processed_products)

        self.cache[cache_key] = filtered_products
        logging.info(f"Found {len(filtered_products)} products after filtering for query: {query}")
        return filtered_products

    def filter_products(self, products):
        filtered = []
        seen_links = set()
        seen_titles_sources = set()
        domain_counts = {domain: 0 for domain in ALLOWED_DOMAINS}
        max_per_source = 3
        total_max = 10

        for p in products:
            if not p or not p.get("link") or not p.get("title") or p["title"] == "Product Title Unavailable":
                continue

            link = p["link"]
            source_domain = extract_domain(link)
            title_source_key = f"{p['title'].lower()}-{p['source'].lower()}"

            if link in seen_links:
                continue
            if title_source_key in seen_titles_sources:
                continue
            if domain_counts.get(source_domain, 0) >= max_per_source:
                continue
            if len(filtered) >= total_max:
                break

            if p['price'] == "Price not available" and p['rating'] == "Rating not available" and len(p['details']) <= 1:
                 logging.info(f"Skipping low-information product: {p['title']}")
                 continue

            filtered.append(p)
            seen_links.add(link)
            seen_titles_sources.add(title_source_key)
            if source_domain in domain_counts:
                 domain_counts[source_domain] += 1

        return filtered

    async def fetch_domain_results(self, params):
        domain = params['q'].split("site:")[-1]
        logging.info(f"Fetching SERP for {domain}...")
        try:
            async with self.session.post("https://google.serper.dev/search", json=params) as res:
                if res.status == 200:
                    results = await res.json()
                    logging.info(f"Got {len(results.get('organic', []))} results from SERP for {domain}")
                    return results.get("organic", [])
                else:
                    logging.error(f"SERP API request failed for {domain} with status {res.status}: {await res.text()}")
                    return []
        except aiohttp.ClientError as e:
            logging.error(f"Network error during SERP API request for {domain}: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error during SERP API request for {domain}: {e}")
            return []

    async def process_serp_results(self, organic_results):
        tasks = []
        valid_links = set()
        for result in organic_results:
            link = result.get("link")
            title = result.get("title", "")
            if not link or not title:
                 continue

            if is_allowed_url(link) and not is_generic_result(title, link) and link not in valid_links:
                 source_domain = extract_domain(link)
                 if self.domain_request_counts.get(source_domain, 0) < self.max_requests_per_domain:
                      self.domain_request_counts[source_domain] = self.domain_request_counts.get(source_domain, 0) + 1
                      tasks.append(self.scrape_product_page(result))
                      valid_links.add(link)
                 else:
                      logging.warning(f"Skipping scrape for {link} due to domain request limit.")

        scraped_products = await asyncio.gather(*tasks)
        return [p for p in scraped_products if p]

    async def scrape_product_page(self, result):
        link = result["link"]
        serp_title = result["title"]
        logging.info(f"Scraping: {link}")
        try:
            await asyncio.sleep(0.5 + self.request_count * 0.1)
            self.request_count += 1

            async with self.session.get(link, timeout=25, allow_redirects=True) as res:
                if res.status != 200:
                    logging.warning(f"Failed to fetch {link} - Status: {res.status}")
                    return None
                try:
                    html_bytes = await res.read()
                    try:
                        html = html_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                         try:
                              html = html_bytes.decode('iso-8859-1')
                         except UnicodeDecodeError:
                              html = html_bytes.decode('cp1252', errors='ignore')

                except Exception as e:
                    logging.error(f"Failed to read/decode HTML from {link}: {e}")
                    return None

                if not html:
                    logging.warning(f"Empty HTML content received from {link}")
                    return None

                soup = BeautifulSoup(html, "lxml")

                if product_unavailable(soup):
                    logging.info(f"Product unavailable: {link}")
                    return None

                domain = extract_domain(link)
                title = clean_title(serp_title, soup)

                if title == "Product Title Unavailable":
                     logging.warning(f"Could not extract a valid title for {link}")
                     return None

                price = extract_price(soup)
                rating = extract_rating(soup, domain)
                details = extract_product_details(soup, domain)
                certifications = extract_certifications(soup)

                if price == "Price not available" and rating == "Rating not available" and len(details) <= 1:
                      logging.info(f"Skipping product due to lack of essential info: {title} ({link})")
                      return None

                product_data = {
                    "title": title,
                    "link": link,
                    "price": price,
                    "rating": rating,
                    "details": details,
                    "certifications": certifications,
                    "source": domain.split('.')[0].title() if '.' in domain else domain.title()
                }
                logging.info(f"Successfully scraped: {title} ({product_data['price']}, {product_data['rating']})")
                return product_data

        except asyncio.TimeoutError:
            logging.warning(f"Timeout scraping {link}")
            return None
        except aiohttp.ClientError as e:
            logging.warning(f"Client error scraping {link}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error scraping {link}: {e}", exc_info=True)
            return None
        finally:
            self.request_count -= 1

async def main():
    st.set_page_config(page_title="MedCompare Pro", page_icon="💊", layout="wide")
    st.title("🌍 MedCompare - Health Supplement Analyzer")

    if 'comparator' not in st.session_state:
        with st.spinner("Initializing medical comparison engine..."):
            st.session_state.comparator = MedComparator()

    comparator = st.session_state.comparator

    with st.form(key="search_form"):
        query = st.text_input(
            "Enter medicine/supplement name or health concern:",
            key="query_input",
            placeholder="E.g.: 'Vitamin C 500mg tablets', 'Probiotics for women', 'Piles relief ayurvedic'"
        )
        submit_button = st.form_submit_button("Search Products")

    if submit_button and query:
        st.session_state.products = None
        st.session_state.analysis = None
        st.session_state.current_query = query

        async with comparator:
            with st.spinner(f"Searching for '{query}'... This may take a minute."):
                adjusted_query = adjust_query(query)
                logging.info(f"Adjusted query: {adjusted_query}")
                products = await comparator.search_products(adjusted_query)
                st.session_state.products = products

    if 'products' in st.session_state and st.session_state.products is not None:
        products = st.session_state.products
        if not products:
            st.error(f"No specific, available products found for '{st.session_state.current_query}'. Try:")
            st.markdown("""
                * Be more specific (e.g., include 'tablets', 'syrup', dosage '500mg', quantity '60 units').
                * Try adding a known brand name (e.g., 'Himalaya Pilex', 'Carbamide Forte').
                * Broaden the search slightly if too specific (e.g., 'best multivitamin' instead of a very niche one).
                * Check your spelling or try different phrasing.
            """)
        else:
            st.markdown(f"Found **{len(products)}** relevant health products for '{st.session_state.current_query}':")
            st.markdown("---")

            for i, product in enumerate(products):
                with st.container():
                    st.subheader(f"{i+1}. {product['title']}")
                    col1, col2, col3 = st.columns([2, 1, 1])
                    col1.markdown(f"**Source:** {product['source']}")
                    col2.markdown(f"**Price:** {product['price']}")
                    col3.markdown(f"**Rating:** {product['rating']}")

                    expander = st.expander("Show Details", expanded=False)
                    with expander:
                         expander.markdown(f"**Details:**")
                         if product['details']:
                             for detail in product['details']:
                                 expander.markdown(f"- {detail}")
                         else:
                             expander.markdown("- No specific details extracted.")

                         if product["certifications"]:
                             expander.markdown("**Certifications:**")
                             for cert in product["certifications"]:
                                 expander.markdown(f"- 🛡️ {cert}")
                         else:
                              expander.markdown("**Certifications:** - None found.")

                         expander.markdown(f"🔗 [View Product Page]({product['link']})", unsafe_allow_html=True)
                    st.markdown("---")

            if products:
                 display_data = [{
                     "Title": p["title"],
                     "Price": p["price"],
                     "Rating": p["rating"],
                     "Details": " | ".join(p["details"][:2]),
                     "Certifications": ", ".join(p["certifications"]),
                     "Source": p["source"],
                     "Link": p["link"]
                 } for p in products]

                 df_display = pd.DataFrame(display_data)

                 st.subheader("Full Product Comparison Table")
                 st.dataframe(df_display, use_container_width=True, column_config={
                     "Link": st.column_config.LinkColumn("View Product", display_text="🔗")
                 })
                 st.markdown("---")

            if products and comparator.groq_client:
                 button_key = "generate_analysis_full_v4"
                 generate_analysis_button = st.button("💡 Generate Recommendation (Fast & Concise)", key=button_key)

                 session_state_key = 'analysis_full_v4'

                 if (session_state_key not in st.session_state or st.session_state[session_state_key] is None) and generate_analysis_button:
                     llm_analysis_data = [{
                         "Title": p["title"],
                         "Price": p["price"],
                         "Rating": p["rating"],
                         "Source": p.get("source", "N/A")
                     } for p in products]

                     if not llm_analysis_data:
                         st.warning("Not enough product data to generate analysis.")
                     else:
                         with st.spinner("💡 Generating concise recommendation (Aiming for < 1 min)..."):
                             prompt_data = "\n".join([
                                 f"- {p['Title']} (Source: {p['Source']}, Price: {p['Price']}, Rating: {p['Rating']})"
                                 for p in llm_analysis_data
                             ])

                             if not prompt_data:
                                 st.error("Could not prepare data for analysis.")
                             else:
                                 user_query = st.session_state.get('current_query', 'health products')
                                 prompt = f"""You are an efficient health product analyst providing a quick recommendation for a user who searched for '{user_query}'.
Review these products based ONLY on Title, Source, Price, and Rating:

{prompt_data}

Instructions:
1. **Directly recommend 1-2 specific products** from the list that seem most suitable for '{user_query}'.
2. **Briefly justify your recommendation (1 sentence per product max)** using Price, Rating, or apparent relevance from the product Title/Source. (Do NOT analyze features you don't have).
3. Keep the *entire response* extremely brief and focused only on the recommendation and its justification (target 2-3 sentences total).
4. Be direct and objective. Do not echo instructions.

Concise Recommendation:
"""
                                 try:
                                     response = comparator.groq_client.chat.completions.create(
                                         messages=[{
                                             "role": "user",
                                             "content": prompt
                                         }],
                                         model="mixtral-8x7b-32768",
                                         max_tokens=200,
                                         temperature=0.6,
                                         stop=["\n\n", "Instructions:", "---", "IMPORTANT:"]
                                     )
                                     analysis_text = response.choices[0].message.content.strip()

                                     if analysis_text and not analysis_text.endswith(('.', '!', '?','.',':')):
                                          last_sentence_end = max(analysis_text.rfind('.'), analysis_text.rfind('?'), analysis_text.rfind('!'))
                                          if last_sentence_end > 0:
                                               analysis_text = analysis_text[:last_sentence_end+1]
                                          else:
                                               analysis_text += '...'

                                     analysis_text += "\n\n**IMPORTANT:** This is an AI-generated recommendation based on limited scraped data (Price, Rating, Title) and NOT medical advice. Always consult a healthcare professional or pharmacist for personalized guidance."
                                     st.session_state[session_state_key] = analysis_text
                                     st.rerun()
                                 except Exception as e:
                                     logging.error(f"Groq API analysis failed: {e}")
                                     st.error(f"Could not generate AI analysis: {e}")
                                     st.session_state[session_state_key] = "Analysis generation failed."

                 if session_state_key in st.session_state and st.session_state[session_state_key]:
                     st.subheader("AI Concise Recommendation")
                     st.markdown(st.session_state[session_state_key])
                     st.markdown("---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.client").setLevel(logging.WARNING)

    if not os.getenv("SERPER_API_KEY"):
        st.error("Fatal Error: SERPER_API_KEY environment variable is not set. The application cannot search for products.")
        st.stop()

    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.error("Application crashed", exc_info=True)