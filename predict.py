# import csv


class Predictor:
    def __init__(self, theta0=0, theta1=0):
        self.theta0 = theta0
        self.theta1 = theta1

    def estimate_price(self, mileage):
        return self.theta0 + (self.theta1 * mileage)


def main():
    try:
        with open('model.txt', 'r') as f:
            data = f.read().strip().split(',')
            theta0, theta1 = [float(value) for value in data]

        predictor = Predictor(theta0, theta1)

        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = predictor.estimate_price(mileage)
        print(f"The estimated price for the car with {mileage} mileage is: {estimated_price}")
    except BaseException as e:
        print(f"Error: ", e)


if __name__ == '__main__':
    main()
