import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from predict import Predictor

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.loss_history = []

    def compute_loss(self, km, price):
        m = len(km)
        errors = Predictor.estimate_price(km, self.theta0, self.theta1) - price
        total_loss = np.sum(errors ** 2) / (2 * m)
        return total_loss

    def gradient_descent(self, km, price):
        m = len(km)
        for _ in range(self.iterations):
            predictions = Predictor.estimate_price(km, self.theta0, self.theta1)
            errors = predictions - price

            tmp_theta0 = self.theta0 - self.learning_rate * np.sum(errors) / m
            tmp_theta1 = self.theta1 - self.learning_rate * np.sum(errors * km) / m

            self.theta0, self.theta1 = tmp_theta0, tmp_theta1

            loss = self.compute_loss(km, price)
            self.loss_history.append(loss)

    def save_coeff(self, filename="coefficient.txt"):
        try:
            with open(filename, "w") as f:
                f.write(f"{self.theta0},{self.theta1}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def plot_graph(self, km, price):
        plt.scatter(km, price, color="blue", label="Data points")
        predicted_prices = Predictor.estimate_price(km, self.theta0, self.theta1)

        plt.plot(km, predicted_prices, color="red", label="Regression line")
        plt.title("Mileage vs Price")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.plot(range(len(self.loss_history)), self.loss_history, color="green")
        plt.title("Loss Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def r_squared(self, km, price):
        predictions = Predictor.estimate_price(km, self.theta0, self.theta1)
        ss_res = np.sum((price - predictions) ** 2)
        ss_tot = np.sum((price - np.mean(price)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def set_theta0(self, theta0):
        self.theta0 = theta0

    def set_theta1(self, theta1):
        self.theta1 = theta1

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def denormalize_theta(theta0, theta1, km_mean, km_std, price_mean, price_std):
    theta1 = theta1 * price_std / km_std
    theta0 = price_mean - theta1 * km_mean
    return theta0, theta1

def main():
    try:
        data = pd.read_csv("data.csv")
        km = data["km"].values
        price = data["price"].values

        km_normalized, km_mean, km_std = normalize_data(km)
        price_normalized, price_mean, price_std = normalize_data(price)

        trainer = LinearRegression()

        trainer.gradient_descent(km_normalized, price_normalized)

        trainer.theta0, trainer.theta1 = denormalize_theta(
            trainer.theta0, trainer.theta1, km_mean, km_std, price_mean, price_std
        )

        print(f"Final theta0: {trainer.theta0}")
        print(f"Final theta1: {trainer.theta1}")
        
        trainer.save_coeff()

        trainer.plot_graph(km, price)
        trainer.plot_loss()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()