import numpy as np

import argparse

from userreg import UserReg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a completed ratings table.")
    parser.add_argument(
        "--name",
        type=str,
        default="ratings_eval.npy",
        help="Name of the npy of the ratings table to complete",
    )

    args = parser.parse_args()

    # Open Ratings table
    print("Ratings loading...")
    table = np.load(args.name)  ## DO NOT CHANGE THIS LINE
    print("Ratings Loaded.")

    print(f"Original shape: {table.shape} (users x movies)")

    model = UserReg(
        k=5,
        lr=0.008,
        lambda_reg=0.02,
        beta_reg=8.0,
        bias_init="medium_adapted_mean",
        num_iterations=30,
    )

    model.fit(table)

    # Predict full matrix
    table = model.predict()
    table = np.clip(table, 0, 5)

    # Save the completed table
    np.save("output.npy", table)  ## DO NOT CHANGE THIS LINE
