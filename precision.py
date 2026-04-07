from train import LinearRegression
from predict import load_coefficients
import pandas as pd


def main():
    """
    Load model coefficients, dataset,
    and compute the R² score.
    """
    try:
        theta0, theta1 = load_coefficients()
        data = pd.read_csv("data.csv")

        if data["km"].isnull().any() or data["price"].isnull().any():
            raise ValueError("Dataset contains NaN values.")

        km = data["km"].values
        price = data["price"].values

        trainer = LinearRegression()
        trainer.set_theta0(theta0)
        trainer.set_theta1(theta1)

        r2 = trainer.r_squared(km, price)

        print(f"R² score: {r2}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()