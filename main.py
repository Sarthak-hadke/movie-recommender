import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Load data
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# User-item matrix
user_item = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# User similarity
user_similarity = cosine_similarity(user_item)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item.index,
    columns=user_item.index
)

# -------------------------------
# Collaborative Filtering
# -------------------------------
def recommend_movies(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    similar_users_movies = user_item.loc[similar_users.index]

    weighted_ratings = similar_users_movies.T.dot(similar_users) / similar_users.sum()

    already_watched = user_item.loc[user_id]
    recommendations = weighted_ratings[already_watched == 0]

    top_movies = recommendations.sort_values(ascending=False).head(n)

    result = pd.DataFrame({
        'movieId': top_movies.index,
        'score': top_movies.values
    })

    result = pd.merge(result, movies, on='movieId')

    return result[['title', 'score']]

# -------------------------------
# Content-Based Filtering
# -------------------------------
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim_movies = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_similar_movies(title):
    indices = pd.Series(movies.index, index=movies['title'])
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim_movies[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][['title']]

# -------------------------------
# Hybrid Recommendation
# -------------------------------
def hybrid_recommendation(user_id, title, n=5):
    collab = recommend_movies(user_id, n=50)

    indices = pd.Series(movies.index, index=movies['title'])
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim_movies[idx]))
    sim_dict = {movies.iloc[i[0]]['title']: i[1] for i in sim_scores}

    collab['content_score'] = collab['title'].map(sim_dict)

    scaler = MinMaxScaler()
    collab[['score', 'content_score']] = scaler.fit_transform(
        collab[['score', 'content_score']].fillna(0)
    )

    collab['final_score'] = 0.6 * collab['score'] + 0.4 * collab['content_score']

    return collab.sort_values('final_score', ascending=False).head(n)

# -------------------------------
# Poster Fetch Function
# -------------------------------
def fetch_poster(movie_title):
    api_key = "40ac659cf0806e36e2147fa187172770"   # 🔥 paste your key

    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"

    response = requests.get(url)
    data = response.json()

    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None