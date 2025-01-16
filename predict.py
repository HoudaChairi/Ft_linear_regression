class Predictor:
    def __init__(self, theta0=0, theta1=0):
        self.theta0 = theta0
        self.theta1 = theta1

    @staticmethod
    def estimate_price(mileage, theta0, theta1):
        """Estimate the price based on mileage and model parameters."""
        return theta0 + (theta1 * mileage)

def main():
    try:
        with open('coefficient.txt', 'r') as f:
            data = f.read().strip().split(',')
            theta0, theta1 = [float(value) for value in data]

        predictor = Predictor(theta0, theta1)

        mileage = float(input("Enter the mileage: "))
        estimated_price = predictor.estimate_price(mileage, theta0, theta1)
        print(f"The estimated price: {estimated_price}")
    except BaseException as e:
        print(f"Error: ", e)


if __name__ == '__main__':
    main()