import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from predict import Predictor


class LinearRegression:
    """
    A class to perform linear regression using gradient descent
    to find the best-fitting line.
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initializes the LinearRegression model with specified
        learning rate and number of iterations.

        learning_rate: The learning rate for gradient descent
        (default is 0.01).
        iterations: The number of iterations for gradient descent
        (default is 1000).
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.loss_history = []

    def calculate_loss(self, errors):
        """
        Computes the loss (mean squared error) of
        the model using precomputed errors.

        errors: Precomputed prediction errors.
        return: The computed loss value.
        """

        m = len(errors)
        total_loss = np.sum(errors ** 2) / (2 * m)
        return total_loss

    def gradient_descent(self, km, price):
        """
        Performs gradient descent to minimize the loss function
        and update model parameters.

        km: The input data for mileage (independent variable).
        price: The actual prices (dependent variable).
        """

        m = len(km)
        for _ in range(self.iterations):
            predict_price = Predictor.estimate_price(
                km, self.theta0, self.theta1)
            errors = predict_price - price

            self.theta0 -= self.learning_rate * np.sum(errors) / m
            self.theta1 -= self.learning_rate * np.sum(errors * km) / m

            loss = self.calculate_loss(errors)
            self.loss_history.append(loss)

    def save_coeff(self, filename="coefficient.txt"):
        """
        Saves the model coefficients (theta0 and theta1) to a file.

        filename: The name of the file to save the coefficients
        (default is 'coefficient.txt').
        """

        try:
            with open(filename, "w") as f:
                f.write(f"{self.theta0},{self.theta1}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def plot_graph(self, km, price):
        """
        Plots a scatter plot of mileage vs. price and the regression line.

        km: The input data for mileage (independent variable).
        price: The actual prices (dependent variable).
        """

        plt.scatter(km, price, color="blue", label="Data points")
        predicted_prices = Predictor.estimate_price(
            km, self.theta0, self.theta1)

        plt.plot(km, predicted_prices, color="red", label="Regression line")
        plt.title("Mileage vs Price")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def plot_loss(self):
        """
        Plots the loss history over iterations.
        """

        plt.plot(range(
            len(self.loss_history)), self.loss_history, color="green")
        plt.title("Loss Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def r_squared(self, km, price):
        """
        Computes the R-squared value to evaluate the model's performance.

        km: The input data for mileage (independent variable).
        price: The actual prices (dependent variable).
        return: The R-squared value.
        """

        predict_price = Predictor.estimate_price(km, self.theta0, self.theta1)
        ss_res = np.sum((price - predict_price) ** 2)
        ss_tot = np.sum((price - np.mean(price)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def set_theta0(self, theta0):
        """
        Sets the value of theta0 (intercept).

        theta0: The intercept value to set.
        """

        self.theta0 = theta0

    def set_theta1(self, theta1):
        """
        Sets the value of theta1 (slope).

        theta1: The slope value to set.
        """

        self.theta1 = theta1


def normalize_data(data):
    """
    Normalizes the input data by subtracting the mean
    and dividing by the standard deviation.

    data: The input data to normalize.
    return: The normalized data, mean, and standard deviation.
    """

    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def denormalize_theta(theta0, theta1, km_mean, km_std, price_mean, price_std):
    """
    Denormalizes the model parameters theta0 and theta1
    using the provided means and standard deviations.

    theta0: The intercept parameter (theta0) of the model.
    theta1: The slope parameter (theta1) of the model.
    km_mean: The mean value of the mileage data.
    km_std: The standard deviation of the mileage data.
    price_mean: The mean value of the price data.
    price_std: The standard deviation of the price data.
    return: The denormalized theta0 and theta1.
    """

    theta1 = theta1 * price_std / km_std
    theta0 = price_mean - theta1 * km_mean
    return theta0, theta1


def main():
    """
    Main function to load data, train the model, and visualize results.
    """

    try:
        data = pd.read_csv("data.csv")
        km = data["km"].values
        price = data["price"].values

        km_normalized, km_mean, km_std = normalize_data(km)
        price_normalized, price_mean, price_std = normalize_data(price)

        trainer = LinearRegression()

        trainer.gradient_descent(km_normalized, price_normalized)

        trainer.theta0, trainer.theta1 = denormalize_theta(
            trainer.theta0, trainer.theta1, km_mean,
            km_std, price_mean, price_std
        )

        print(f"theta0: {trainer.theta0}")
        print(f"theta1: {trainer.theta1}")

        trainer.save_coeff()

        trainer.plot_graph(km, price)
        trainer.plot_loss()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
