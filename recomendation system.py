import numpy as np

# Sample user-item matrix (rows represent users, columns represent movies)
# Replace this with your actual data or a larger dataset
user_item_matrix = np.array([
    [5, 4, 0, 1, 0],
    [0, 0, 5, 2, 3],
    [2, 0, 4, 0, 5],
    [0, 3, 0, 4, 4],
    [4, 0, 3, 0, 0]
])

# Function to recommend movies to a user based on collaborative filtering
def collaborative_filtering_recommendation(user_item_matrix, user_id, num_recommendations=3):
    user_ratings = user_item_matrix[user_id]

    # Find similar users using cosine similarity
    similarities = np.dot(user_item_matrix, user_ratings) / (np.linalg.norm(user_item_matrix, axis=1) * np.linalg.norm(user_ratings))
    
    # Sort users by similarity in descending order
    similar_users = np.argsort(similarities)[::-1]

    # Exclude the user's own id
    similar_users = similar_users[similar_users != user_id]

    # Find movies that the user hasn't rated yet
    unrated_movies = np.where(user_ratings == 0)[0]

    # Recommend movies based on ratings from similar users
    recommendations = []
    for movie_id in unrated_movies:
        weighted_sum = 0
        similarity_sum = 0
        for similar_user in similar_users:
            if user_item_matrix[similar_user, movie_id] != 0:
                weighted_sum += similarities[similar_user] * user_item_matrix[similar_user, movie_id]
                similarity_sum += np.abs(similarities[similar_user])
        if similarity_sum != 0:
            predicted_rating = weighted_sum / similarity_sum
            recommendations.append((movie_id, predicted_rating))

    # Sort recommendations by predicted rating in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommendations
    top_recommendations = recommendations[:num_recommendations]

    return top_recommendations

# Example usage
user_id = 0
recommendations = collaborative_filtering_recommendation(user_item_matrix, user_id)
print(f"Top 3 movie recommendations for user {user_id}:")
for movie_id, predicted_rating in recommendations:
    print(f"Movie {movie_id + 1}: Predicted Rating - {predicted_rating}")
