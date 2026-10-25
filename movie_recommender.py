"""
Movie Recommendation System (Content-Based, using AI/ML techniques)
Recommends movies similar to a given movie based on genre/description
similarity using TF-IDF vectorization and cosine similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset (in a real project, load from a CSV of thousands of movies)
movies = pd.DataFrame({
    "title": [
        "Inception", "Interstellar", "The Dark Knight",
        "Toy Story", "Finding Nemo", "The Avengers",
        "Iron Man", "Up", "The Matrix", "John Wick"
    ],
    "genres": [
        "sci-fi thriller mind-bending", "sci-fi space drama",
        "action crime superhero", "animation family adventure",
        "animation family ocean", "action superhero team",
        "action superhero tech", "animation family adventure",
        "sci-fi action dystopian", "action thriller revenge"
    ]
})


class MovieRecommender:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(data["genres"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_title: str, top_n: int = 3):
        if movie_title not in self.data["title"].values:
            return f"'{movie_title}' not found in dataset."

        idx = self.data[self.data["title"] == movie_title].index[0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        recommendations = [self.data.iloc[i]["title"] for i, _ in scores]
        return recommendations


if __name__ == "__main__":
    recommender = MovieRecommender(movies)

    print("=== Movie Recommendation System ===\n")
    movie = input("Enter a movie you like: ").strip()

    results = recommender.recommend(movie)
    if isinstance(results, str):
        print(results)
    else:
        print(f"\nBecause you liked '{movie}', you might also enjoy:")
        for i, title in enumerate(results, 1):
            print(f"{i}. {title}")
