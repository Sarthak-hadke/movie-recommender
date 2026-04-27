import streamlit as st
from main import hybrid_recommendation, movies, fetch_poster

# -------------------------------
# 🔥 TOP (CONFIG + TITLE)
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie recommendations using AI")

# -------------------------------
# 🎯 INPUTS
# -------------------------------
user_id = st.number_input("User ID", min_value=1, value=1)

movie_list = movies['title'].sort_values().unique()
selected_movie = st.selectbox("Pick a movie you like", movie_list)

# -------------------------------
# 🚀 BUTTON + RECOMMENDATION
# -------------------------------
if st.button("Recommend"):
    with st.spinner("Finding best movies for you..."):
        result = hybrid_recommendation(user_id, selected_movie)

    # ❗ Handle empty case
    if result.empty:
        st.warning("No recommendations found. Try another movie.")
    else:
        st.subheader("🔥 Recommended Movies")

        # -------------------------------
        # 🎬 GRID LAYOUT (ADD HERE)
        # -------------------------------
        cols = st.columns(5)

        for i, row in result.iterrows():
            col = cols[i % 5]

            poster = fetch_poster(row['title'])

            with col:
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.write("🎬 No Image")

                st.caption(row['title'])
                st.caption(f"⭐ {round(row['final_score'], 2)}")