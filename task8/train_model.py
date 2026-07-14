import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("ml-latest-small/movies.csv")

# Fill missing values
movies["genres"] = movies["genres"].fillna("")

# Convert genres into vectors
cv = CountVectorizer(stop_words="english")

genre_matrix = cv.fit_transform(movies["genres"])

# Calculate similarity matrix
similarity = cosine_similarity(genre_matrix)

# Save movie titles
pickle.dump(movies, open("movie_list.pkl", "wb"))

# Save similarity matrix
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Model Created Successfully!")