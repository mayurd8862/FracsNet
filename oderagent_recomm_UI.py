import streamlit as st
import asyncio
from comparison import MedComparator
import pandas as pd
import logging

async def main():
    st.set_page_config(page_title="MedCompare Pro", page_icon="💊", layout="centered")
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
                products = await comparator.search_products(query)
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
    asyncio.run(main())