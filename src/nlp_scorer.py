import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NLPScorer:
    def __init__(self, data_path="data/mock_ads.csv"):
        try:
            self.df = pd.read_csv(data_path, usecols=['product_name', 'description'])
            self.df = self.df.dropna().reset_index(drop=True)
            
            if len(self.df) > 5000:
                self.df = self.df.sample(5000, random_state=42).reset_index(drop=True)
        except FileNotFoundError:
            raise Exception("mock_ads.csv not found in the data folder.")

        # TF-IDF
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['description'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_similarity(self, product_a, product_b):
        """TF-IDF cosine similarity"""
        try:
            idx_a = self.df.index[self.df['product_name'] == product_a].tolist()[0]
            idx_b = self.df.index[self.df['product_name'] == product_b].tolist()[0]
            return self.cosine_sim[idx_a][idx_b]
        except IndexError:
            return 0.0

    def get_candidates(self, current_product, n=3):
        """top N most similar products """
        try:
            idx = self.df.index[self.df['product_name'] == current_product].tolist()[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Skip the first one (itself)
            candidate_indices = [i[0] for i in sim_scores[1:n+1]]
            return self.df['product_name'].iloc[candidate_indices].tolist()
        except IndexError:
            return []
            
    def get_random_product(self):
        return self.df['product_name'].sample().iloc[0]