
class Predictor:
    """
    A class to predict the price based on mileage
    and model parameters (theta0 and theta1).
    """

    def __init__(self, theta0=0, theta1=0):
        """
        Initializes the Predictor with given model parameters (theta0, theta1).

        param theta0: The intercept parameter of the model (default is 0).
        param theta1: The slope parameter of the model (default is 0).
        """
        self.theta0 = theta0
        self.theta1 = theta1

    @staticmethod
    def estimate_price(mileage, theta0, theta1):
        """
        Estimates the price based on mileage and model parameters.

        :param mileage: The mileage of the car.
        :param theta0: The intercept parameter of the model.
        :param theta1: The slope parameter of the model.
        :return: The estimated price.
        """
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
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
