# 1. Upload the ml-100k.zip dataset
from google.colab import files
uploaded = files.upload()  # Select and upload `ml-100k.zip`

# 2. Reinstall numpy and scikit-surprise for compatibility
# Run these in a clean environment if you run into version mismatches
!pip uninstall -y numpy scikit-surprise
!pip install numpy==1.23.5
!pip install scikit-surprise

# 3. Load and merge the data
import pandas as pd

# Load ratings data
ratings = pd.read_csv("u.data", sep='\t', names=["user_id", "movie_id", "rating", "timestamp"])

# Load movie titles
movies = pd.read_csv("u.item", sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=["movie_id", "title"])

# Merge ratings with movie titles
df = pd.merge(ratings, movies, on="movie_id")
df.head()

# 4. Hyperparameter tuning using GridSearchCV
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

# Define rating scale and format for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

# Define the parameter grid to search
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.002, 0.005], 'reg_all': [0.4, 0.6]}

# Perform GridSearchCV
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)

# Print the best RMSE score and parameters
print(f"Best RMSE score: {gs.best_score['rmse']}")
print(f"Best parameters: {gs.best_params['rmse']}")

# 5. Train and evaluate the final SVD model
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use the SVD model with the best parameters found
best_params = gs.best_params['rmse']
model = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate performance on the test set
print("Model evaluation on the test set:")
accuracy.rmse(predictions)
accuracy.mae(predictions)

# 6. Generate Top-N Movie Recommendations
def get_top_n(predictions, n=5):
    """Return the top-N recommendation for each user from a set of predictions."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
        
    return top_n

# Get top 5 movies recommended for all users in the test set
top_n_recommendations = get_top_n(predictions, n=5)

# Print recommendations for a few users (e.g., the first 5 users in the test set)
print("\nTop 5 recommendations for a few users in the test set:")
for i, (user_id, user_recs) in enumerate(list(top_n_recommendations.items())[:5]):
    print(f"\nRecommendations for User {user_id}:")
    for movie_id, score in user_recs:
        # Get the movie title from the movies DataFrame
        title = movies[movies["movie_id"] == int(movie_id)]["title"].values[0]
        print(f"- {title} (Predicted rating: {score:.2f})")
