# 🎬 Movie Recommendation System

A hybrid movie recommendation system built using Machine Learning that suggests movies based on user preferences.
This project combines Collaborative Filtering and Content-Based Filtering to provide accurate recommendations.

---

## 🚀 Live Demo

👉 https://movie-recommender-wu2ud69iymnqhktq4xbxuy.streamlit.app/

---

##  Features

* Personalized recommendations using user ratings
* Hybrid model (Collaborative + Content-Based)
* Movie posters using TMDB API
* Interactive UI using Streamlit

---

##  How It Works

### Collaborative Filtering

* Finds similar users
* Recommends movies liked by them

### Content-Based Filtering

* Uses genres
* Finds similar movies

### Hybrid Model

Final Score = 0.6 × Collaborative + 0.4 × Content

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TMDB API

---

##  Project Structure

movie-recommender/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── app.py
├── main.py
├── requirements.txt
└── README.md

---

## Run Locally

git clone https://github.com/Sarthak-hadke
/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
streamlit run app.py

---

## Author

Sarthak Hadke

# 🎬 Movie Recommendation System

This is my ML project deployed using Streamlit.
