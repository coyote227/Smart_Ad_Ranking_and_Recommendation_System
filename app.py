import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from src.nlp_scorer import NLPScorer


st.set_page_config(page_title="Ad Recommender System", layout="centered")

@st.cache_resource
def load_model():
    return NLPScorer("data/mock_ads.csv")

try:
    with st.spinner("Training TF-IDF Model"):
        scorer = load_model()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.title("Smart Ad Recommendation System")
st.markdown("""
This system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to recommend the most contextually relevant products based on their descriptions.
""")
st.divider()


st.subheader("Select a Product to View")

# Create a searchable dropdown of all available products
all_products = scorer.df['product_name'].tolist()
selected_product = st.selectbox("Search for a product:", all_products)

# Allow the user to choose how many recommendations they want
top_n = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=5)

st.divider()

# Recommendation Logic
if selected_product:
    st.subheader(f"Because you viewed: *{selected_product}*")
    
    with st.spinner("Finding similar items"):
        # We can use the logic inside your nlp_scorer to find the best matches
        idx = scorer.df.index[scorer.df['product_name'] == selected_product].tolist()[0]
        
        # Get pairwise similarity scores for the selected product
        sim_scores = list(enumerate(scorer.cosine_sim[idx]))
        
        # Sort them in descending order (highest similarity first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Grab the top N (skipping the first one, which is the product itself)
        top_matches = sim_scores[1:top_n+1]
        
        if not top_matches:
            st.info("No recommendations found.")
        else:
            # Display the recommendations in a clean list
            for i, match in enumerate(top_matches):
                match_idx = match[0]
                similarity_score = match[1] * 100
                recommended_name = scorer.df['product_name'].iloc[match_idx]
                
                # Create a visual card for each recommendation
                with st.container():
                    st.markdown(f"#### {i+1}. {recommended_name}")
                    st.progress(min(match[1], 1.0), text=f"Match Score: {similarity_score:.1f}%")
                    st.write("") # small spacer