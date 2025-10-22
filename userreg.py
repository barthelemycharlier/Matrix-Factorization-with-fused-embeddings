import numpy as np
from tqdm import trange


class UserReg:
    """
    UserReg: Matrix Factorization with User Regularization
    """

    def __init__(
        self,
        k=5,
        lr=0.01,
        lambda_reg=0.05,
        beta_reg=8.0,
        bias_init="medium_adapted_mean",
        num_iterations=30,
        random_state=42,
        verbose=True,
    ):
        self.k = k
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg
        self.bias_init = bias_init
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, R):
        """
        Train the model using the observed ratings matrix R (users × items).
        Missing entries should be np.nan.
        """
        np.random.seed(self.random_state)
        n_users, n_items = R.shape

        # Parameters
        av_rating = np.nanmean(R)
        self.P = 0.1 * np.random.randn(n_users, self.k)
        self.Q = 0.1 * np.random.randn(n_items, self.k)

        # Initialize biases with a heuristic
        av_per_user = np.nanmean(R, axis=1)
        av_per_item = np.nanmean(R, axis=0)

        if self.bias_init == "global_mean":
            self.bu = np.ones(n_users) * av_rating
            self.bi = np.ones(n_items) * av_rating
        elif self.bias_init == "none":
            self.bu = np.zeros(n_users)
            self.bi = np.zeros(n_items)
        else:
            av_per_user = np.nanmean(R, axis=1)
            av_per_item = np.nanmean(R, axis=0)
            if self.bias_init == "strong_adapted_mean":
                self.bu = (
                    np.ones(n_users) * av_rating
                    + (np.ones(n_users) * av_rating - av_per_user) / 4
                ) / 2.2
                self.bi = (
                    np.ones(n_items) * av_rating
                    + (np.ones(n_items) * av_rating - av_per_item) / 4
                ) / 2.2
            elif self.bias_init == "medium_adapted_mean":
                self.bu = (
                    np.ones(n_users) * av_rating
                    + (np.ones(n_users) * av_rating - av_per_user) / 4
                ) / 2.4
                self.bi = (
                    np.ones(n_items) * av_rating
                    + (np.ones(n_items) * av_rating - av_per_item) / 4
                ) / 2.4
            elif self.bias_init == "weak_adapted_mean":
                self.bu = (
                    np.ones(n_users) * av_rating
                    + (np.ones(n_users) * av_rating - av_per_user) / 4
                ) / 2.6
                self.bi = (
                    np.ones(n_items) * av_rating
                    + (np.ones(n_items) * av_rating - av_per_item) / 4
                ) / 2.6

        mask = ~np.isnan(R)
        R_filled = np.nan_to_num(R, nan=0.0)

        iterator = (
            trange(self.num_iterations, desc="Training UserReg")
            if self.verbose
            else range(self.num_iterations)
        )

        for _ in iterator:
            for u in range(n_users):
                Iu = np.where((mask[u, :]) & (R[u, :] > 3))[0]
                if len(Iu) == 0:
                    continue

                mean_Q = np.mean(self.Q[Iu, :], axis=0)
                for i in Iu:
                    e_ui = R_filled[u, i] - (
                        self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                    )

                    # Update biases
                    self.bu[u] += self.lr * (e_ui - self.lambda_reg * self.bu[u])
                    self.bi[i] += self.lr * (e_ui - self.lambda_reg * self.bi[i])

                    # Update latent factors
                    self.P[u] += self.lr * (
                        e_ui * self.Q[i]
                        - self.beta_reg * (self.P[u] - mean_Q)
                        - self.lambda_reg * self.P[u]
                    )
                    self.Q[i] += self.lr * (
                        e_ui * self.P[u] - self.lambda_reg * self.Q[i]
                    )

        return self

    def predict(self):
        """Return the full predicted ratings matrix."""
        preds = self.P @ self.Q.T + self.bu[:, None] + self.bi[None, :]
        return preds

    def predict_user(self, u):
        """Predict ratings for a single user index u."""
        preds = self.P[u] @ self.Q.T + self.bu[u] + self.bi
        return preds

    def constrain_ratings(self, predictions):
        """Round ratings to [0,5] in 0.5 increments."""
        return np.round(predictions * 2) / 2



class WUserReg:
    """
    UserReg: Matrix Factorization with User Regularization
    """

    def __init__(
        self,
        k=5,
        lr=0.01,
        lambda_reg=0.05,
        beta_reg=8.0,
        bias_init="medium_adapted_mean",
        num_iterations=30,
        random_state=42,
        verbose=True,
        alpha1=0.2,
        alpha2=0.2,
        m1=10,
        m2=10,
    ):
        self.k = k
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg
        self.bias_init = bias_init
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.verbose = verbose
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.m1 = m1
        self.m2 = m2


    def fit(self, R):
        """
        Train the model using the observed ratings matrix R (users × items).
        Missing entries should be np.nan.
        """
        np.random.seed(self.random_state)
        n_users, n_items = R.shape

        # Parameters
        av_rating = np.nanmean(R)
        self.P = 0.1 * np.random.randn(n_users, self.k)
        self.Q = 0.1 * np.random.randn(n_items, self.k)

        # Initialize user's and movie's biases
        av_per_user = np.nanmean(R, axis=1)
        av_per_item = np.nanmean(R, axis=0)

        if self.bias_init == "global_mean":
            self.bu = np.ones(n_users) * av_rating
            self.bi = np.ones(n_items) * av_rating
        elif self.bias_init == "none":
            self.bu = np.zeros(n_users)
            self.bi = np.zeros(n_items)
        else:
            av_per_user = np.nanmean(R, axis=1)
            av_per_item = np.nanmean(R, axis=0)
            if self.bias_init == "strong_adapted_mean":
                self.bu = (
                    np.ones(n_users) * av_rating
                    + (np.ones(n_users) * av_rating - av_per_user) / 4
                ) / 2.2
                self.bi = (
                    np.ones(n_items) * av_rating
                    + (np.ones(n_items) * av_rating - av_per_item) / 4
                ) / 2.2
            elif self.bias_init == "medium_adapted_mean":
                self.bu = (
                    np.ones(n_users) * av_rating
                    + (np.ones(n_users) * av_rating - av_per_user) / 4
                ) / 2.4
                self.bi = (
                    np.ones(n_items) * av_rating
                    + (np.ones(n_items) * av_rating - av_per_item) / 4
                ) / 2.4
            elif self.bias_init == "weak_adapted_mean":
                self.bu = (
                    np.ones(n_users) * av_rating
                    + (np.ones(n_users) * av_rating - av_per_user) / 4
                ) / 2.6
                self.bi = (
                    np.ones(n_items) * av_rating
                    + (np.ones(n_items) * av_rating - av_per_item) / 4
                ) / 2.6

        mask = ~np.isnan(R)
        R_filled = np.nan_to_num(R, nan=0.0)

        weight_matrix = self.calculate_normalized_weights(R_filled)

        iterator = (
            trange(self.num_iterations, desc="Training UserReg")
            if self.verbose
            else range(self.num_iterations)
        )

        for _ in iterator:
            for u in range(n_users):
                Iu = np.where(mask[u, :])[0]
                if len(Iu) == 0:
                    continue

                mean_Q = np.mean(self.Q[Iu, :], axis=0)
                for i in Iu:
                    e_ui = [(R_filled[u, i] - (
                        self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])) * (weight_matrix[u, i])
                    )] 

                    # Update biases
                    self.bu[u] += self.lr * (e_ui - self.lambda_reg * self.bu[u])
                    self.bi[i] += self.lr * (e_ui - self.lambda_reg * self.bi[i])

                    # Update latent factors
                    self.P[u] += self.lr * (
                        e_ui * self.Q[i]
                        - self.beta_reg * (self.P[u] - mean_Q)
                        - self.lambda_reg * self.P[u]
                    )
                    self.Q[i] += self.lr * (
                        e_ui * self.P[u] - self.lambda_reg * self.Q[i]
                    )

        return self

    def calculate_normalized_weights(self, ratings_matrix):
        n_users, n_items = ratings_matrix.shape

        #Get user activity and item popularity
        user_activity = np.count_nonzero(ratings_matrix, axis=1)
        item_popularity = np.count_nonzero(ratings_matrix, axis=0)

        user_indices, item_indices = np.nonzero(ratings_matrix)

        if len(user_indices) == 0:
            print("Warning: No ratings found in matrix.")
            return np.zeros_like(ratings_matrix)

        print(f"Found {len(user_indices)} total ratings.")

        #Calculate un-normalized weights 
        N_u_vec = user_activity[user_indices].astype(float)
        M_i_vec = item_popularity[item_indices].astype(float)

        #Add smoothing
        user_term = np.power(N_u_vec + self.m1, self.alpha1)
        item_term = np.power(M_i_vec + self.m2, self.alpha2)
        
        w_prime_vec = 1.0 / (user_term * item_term)

        w_prime_avg = np.mean(w_prime_vec)
        print(f"Average un-normalized weight (w'_avg): {w_prime_avg:.4f}")
        final_weights_vec = w_prime_vec / w_prime_avg
        

        w = np.zeros((n_users, n_items), dtype=float)
        w[user_indices, item_indices] = final_weights_vec
        
        return w

    def predict(self):
        """Return the full predicted ratings matrix."""
        preds = self.P @ self.Q.T + self.bu[:, None] + self.bi[None, :]
        return preds

    def predict_user(self, u):
        """Predict ratings for a single user index u."""
        preds = self.P[u] @ self.Q.T + self.bu[u] + self.bi
        return preds

    def constrain_ratings(self, predictions):
        """Round ratings to [0,5] in 0.5 increments."""
        return np.round(predictions * 2) / 2
