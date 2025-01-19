from train import LinearRegression
import pandas as pd


def main():
    """
    Main function to load model coefficients,
    input data, and calculate the R² score.
    """
    try:
        with open('coefficient.txt', 'r') as f:
            data = f.read().strip().split(',')
            theta0, theta1 = [float(value) for value in data]

        data = pd.read_csv("data.csv")
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
