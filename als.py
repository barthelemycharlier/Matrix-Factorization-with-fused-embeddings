import pandas as pd
import numpy as np
from tqdm import trange
from scipy.optimize import lsq_linear
def annls_optimization_bounded(ratings_matrix, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=1):
    """
    ALS optimization for matrix factorization with bounded NNLS
    
    Matrix dimensions:
    - R: 610x4980 (users x movies) - input format
    - movie_factors: 4980xk (movies x k factors)
    - user_factors: 610xk (users x k factors)
    - Reconstruction: user_factors @ movie_factors.T = 610x4980 (users x movies)
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))  # 4980×k
    user_factors = np.random.normal(0, 0.1, (n_users, k))    # 610×k
    
    # Create mask for observed ratings
    mask_original = ~np.isnan(R_original)  # (610, 4980) users × movies
    R_clean = np.nan_to_num(R_original, nan=0.0)
    
    # Set bounds for factors - keeps product in reasonable range
    # If each factor is bounded by max_factor, product is bounded by k * max_factor^2
    max_factor = np.sqrt(5 / k)  # This ensures reconstruction stays near [0,5]
    
    for iteration in range(num_iterations):
        # For each movie, find users who rated it and update its factors
        for movie_idx in range(movie_factors.shape[0]):  # 0 to 4979
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                U_items = user_factors[user_indices, :]
                ratings_i = R_clean[user_indices, movie_idx]
                
                # Set up regularized system
                A = np.vstack([U_items, np.sqrt(lambda_reg) * np.eye(k)])
                b = np.hstack([ratings_i, np.zeros(k)])
                
                # Use bounded least squares: factors bounded in [0, max_factor]
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                movie_factors[movie_idx, :] = result.x
        
        # For each user, find movies they rated and update their factors
        for user_idx in range(user_factors.shape[0]):  # 0 to 609
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                I_users = movie_factors[movie_indices, :]
                ratings_u = R_clean[user_idx, movie_indices]
                
                # Set up regularized system
                A = np.vstack([I_users, np.sqrt(mu_reg) * np.eye(k)])
                b = np.hstack([ratings_u, np.zeros(k)])
                
                # Use bounded least squares: factors bounded in [0, max_factor]
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                user_factors[user_idx, :] = result.x
    
    # Reconstruct and clip: user_factors @ movie_factors.T gives (610×4980)
    predictions = user_factors @ movie_factors.T
    return np.clip(predictions, 0, 5)

def constrain_ratings(predictions):
    """Constrain predictions to valid movie ratings: 0-5 in 0.5 increments"""
    return np.round(predictions * 2) / 2

# Apply ALS matrix factorization with bounded NNLS
# print("Applying ANNLS matrix factorization with bounded optimization...")
# k = 1
# predicted_ratings = annls_optimization_bounded(table, k=k, num_iterations=30, lambda_reg=0.01, mu_reg=1)
# table = constrain_ratings(predicted_ratings)
# print("ANNLS optimization with bounded factors completed.")

# print(f"Final shape: {table.shape} (users x movies)")