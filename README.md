# Smart Ad Recommendation System

## Overview

This project implements a content-based recommendation system using **TF-IDF** and **Cosine Similarity** to suggest similar products based on their descriptions.

Built with **Streamlit** for an interactive UI.

---

## Features

* Product-based recommendation system
* TF-IDF vectorization of product descriptions
* Cosine similarity ranking
* Interactive UI with search and ranking display

---

## Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn

---

## Project Structure

```
Smart_Ad_Ranking_and_Recommendation_System/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── mock_ads.csv
├── src/
│   ├── __init__.py
│   └── nlp_scorer.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## How It Works

* Product descriptions are converted into TF-IDF vectors
* Cosine similarity is computed between all products
* Top-N similar products are recommended

---

## Future Improvements

* Hybrid recommendation system (content + collaborative)
* Deployment (Streamlit Cloud / Docker)
* Real-time user interaction tracking