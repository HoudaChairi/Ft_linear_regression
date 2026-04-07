
class Predictor:
    """
    A class to predict the price based on mileage
    and model parameters (theta0 and theta1).
    """

    def __init__(self, theta0=0, theta1=0):
        self.theta0 = theta0
        self.theta1 = theta1

    def predict(self, mileage):
        return self.theta0 + (self.theta1 * mileage)

    @staticmethod
    def estimate_price(mileage, theta0, theta1):
        return theta0 + (theta1 * mileage)


def load_coefficients(filename="coefficient.txt"):
    """Load theta0 and theta1 safely from file."""
    try:
        with open(filename, 'r') as f:
            data = f.read().strip().split(',')

            if len(data) < 2:
                raise ValueError("Not enough coefficients.")

            theta0, theta1 = [float(value.strip()) for value in data[:2]]
            return theta0, theta1

    except (FileNotFoundError, ValueError):
        return 0, 0


def get_valid_mileage():
    """Validate user input."""
    while True:
        try:
            user_input = input("Enter the mileage: ").strip()

            if not user_input:
                print("Mileage cannot be empty.")
                continue

            mileage = float(user_input)

            if mileage < 0:
                print("Mileage cannot be negative.")
                continue

            return mileage

        except ValueError:
            print("Please enter a valid number.")

        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting cleanly.")
            exit()

def main():
    theta0, theta1 = load_coefficients()

    predictor = Predictor(theta0, theta1)

    mileage = get_valid_mileage()

    estimated_price = predictor.predict(mileage)
    print(f"The estimated price: {estimated_price}")


if __name__ == '__main__':
    main()