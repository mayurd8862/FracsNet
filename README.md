# 🏥 Medimate : Healthcare E-Commerce Multi-Agent System

A modular, agent-driven platform designed to handle a wide range of user queries for healthcare products using a combination of rule-based logic, retrieval-augmented generation (RAG), and multi-agent coordination.

## 🚀 Overview

This system is built to support a healthcare e-commerce chatbot with intelligent, autonomous agents that:

- 🔁 Multi-Agent Architecture with clearly defined responsibilities
- 🔍 Query Classification through hybrid intent detection
- 🧠 Retrieval-Augmented Generation (RAG) with vector DB
- 🛍️ Product Comparison Engine using web scraping (BeautifulSoup + Serper API)
- 📦 Order Management with MongoDB backend
- 📝 Summarization Agent with extractive + abstractive methods
- 📤 Email Notifications for order confirmation
- 🎯 Product recommendation according to the content 

# 📐 System Architecture

![image](https://github.com/user-attachments/assets/f4915396-69ab-4efc-a7b7-03bd9c35172a)


## 🧠 Agent Details
### 1. ✅ Query Validator Agent
- Checks for malformed inputs and invalid syntax
- Prompts user to resubmit if query is unclear

### 2. 🧭 Router Agent
Routes query to one of the 5 agents using:
- Keyword matching
- Sentence structure
- Embedding-based similarity
- Rule-based intent classifiers

### 3. 📚 General Info Agent (RAG)
- Uses vector DB to search embedded medical product data
- Employs RAG pipeline:

🔎 Retriever: Pulls relevant product chunks
🧾 Generator: LLM (e.g., GPT) answers grounded in retrieved context

#### 4. ⚖️ Comparison Agent
- Finds and compares health product listings from ALLOWED_DOMAINS
Cleans data using:
- Extracts: name, price, rating, certifications, description

#### 5. ✂️ Summarizer Agent
- Extracts key points
- Uses hybrid extractive + abstractive summarization

#### 6. 📦 Order Agent
- Manages:
New orders, Previous orders

- Integrates with:
MongoDB Atlas for storage, Email system for notifications

- Offers personalized recommendations based on:
Past orders, Popular and complementary products









