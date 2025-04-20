import pandas as pd

movies = pd.read_csv('./dataset/tmdb_5000_movies.csv')
print("Columns:", movies.columns.tolist())
print("\nSample release_date values:")
print(movies[["id", "title", "release_date"]].head(10))
print("\nMissing release_date count:", movies["release_date"].isna().sum())