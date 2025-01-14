import csv


class Predictor:
    def __init__(self, theta0=0, theta1=0):
        self.theta0 = theta0
        self.theta1 = theta1

    def estimate_price(self, mileage):
        return self.theta0 + (self.theta1 * mileage)


def main():
   


if __name__ == '__main__':
    main()
