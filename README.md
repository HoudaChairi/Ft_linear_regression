# Ft_linear_regression 📈🚗

## 📝 Description
This project implements a simple linear regression model to predict car prices based on mileage. It serves as an introduction to machine learning fundamentals, focusing on:
- Understanding linear regression with a single feature
- Implementation of gradient descent algorithm
- Data visualization and model evaluation
- Data scaling (normalization and denormalization)

<!---## Plot Data after Training -->

  ![scree3](https://github.com/user-attachments/assets/19939eb0-8419-4980-958f-125afcbe6b75)

---

## 🧠 Concepts Needed for the Project
* Linear Regression 📈
  
  A fundamental statistical and machine learning approach that models the relationship between a dependent variable (target)
    and one or more independent variables (features) by fitting a linear equation to the observed data.
  - Understanding the relationship between independent and dependent variables
  - How to fit a line to data points
  - Predicting continuous values based on input features

* Gradient Descent 📉
  
  An optimization algorithm that iteratively adjusts parameters to minimize an error function by computing
    the gradient (derivative) of the loss function and moving in the direction of steepest descent.
  - Iterative optimization algorithm
  - Finding the minimum of the cost function
  - Updating parameters to improve predictions

* Loss Function 🧮
  
  A function that measures how well your model's predictions match the actual data by calculating the difference
    between predicted and actual values, where a smaller value indicates better model performance.
  - Measuring prediction errors
  - Mean Squared Error (MSE)
  - Cost function optimization

* Feature Scaling 🔄

  Normalizing the features is important for gradient descent to converge faster and help the algorithm perform better by preventing one feature from dominating the others.
  - **Normalization** (Min-Max Scaling) for features to a range [0, 1]
  - **Denormalization** for converting the scaled features back to their original values
  
  Scaling is applied to the feature before fitting the model and denormalization is used to convert predictions back to the original scale.

---

### 🧮 Linear Regression Equation

The price prediction is based on the following hypothesis:

$$
\text{estimatePrice}(mileage) = \theta_0 + (\theta_1 \times mileage)
$$

Where:
- `θ₀` (theta0): Y-intercept
- `θ₁` (theta1): Slope of the line
- `mileage`: Input feature (X variable)

---

### 🚀 Gradient Descent Update Rules 

The model uses gradient descent to minimize the cost function with these update rules:

$$
\text{tmp}\theta_0 = \text{learningRate} \times \left( \frac{1}{m} \right) \times \sum \left( \text{estimatePrice}(mileage[i]) - price[i] \right)
$$

$$
\text{tmp}\theta_1 = \text{learningRate} \times \left( \frac{1}{m} \right) \times \sum \left( \left( \text{estimatePrice}(mileage[i]) - price[i] \right) \times mileage[i] \right)
$$

---

### 📉 Mean Squared Error (MSE)

The `Mean Squared Error (MSE)` is a metric used to evaluate the performance of the linear regression model
by quantifying the average squared difference between predicted values and actual values.
A lower MSE indicates that the model's predictions are closer to the actual data.

The equation for MSE is as follows:

$$
\text{MSE} = \left( \frac{1}{m} \right) \times \sum \left( \left( \text{actual}[i] - \text{predicted}[i] \right)^2 \right)
$$

Where:
- `m` is the total number of data points
- `actual[i]` is the actual value for the i-th data point
- `predicted[i]` is the predicted value for the i-th data point

---

### 📊 R-squared (R²)

The `R-squared` value is a statistical measure that indicates the proportion of variance in the dependent variable explained by the independent variable(s). It provides insight into the goodness of fit of the model, with values ranging from 0 to 1. A higher R-squared value indicates a better fit.

The equation for R-squared is as follows:

$$
R^2 = 1 - \left( \frac{SS_{\text{res}}}{SS_{\text{tot}}} \right)
$$

#### Sum of Squares

$$
SS_{\text{res}} = \sum_{i=1}^{m} \left( \text{actual}[i] - \text{predicted}[i] \right)^2
$$

$$
SS_{\text{tot}} = \sum_{i=1}^{m} \left( \text{actual}[i] - \text{mean}(\text{actual}) \right)^2
$$

Where:
- **actual[i]** is the actual value for the i-th data point
- **predicted[i]** is the predicted value for the i-th data point
- **mean(actual)** is the mean of all actual values

---

## 🛠️ Project Structure
The project consists of two main programs:
1. **Price Predictor**
   - Prompts user for mileage input
   - Returns estimated car price using trained parameters
   - Uses saved θ₀ and θ₁ values

2. **Model Trainer**
   - Reads dataset of car prices and mileages
   - Implements gradient descent algorithm
   - Saves optimized θ₀ and θ₁ values
   - Visualizes data and regression line (bonus feature)

---

## 📊 Bonus Features
- **Data Visualization**: Plot showing:
  - Raw data points (mileage vs. price)
  - Fitted regression line
  - Interactive visualization capabilities
- **Model Evaluation**: Program calculating:
  - Mean Squared Error (MSE)
  - R-squared value
  - Prediction accuracy metrics

## 📈 Learning Outcomes
Through this project, you will learn:
- Fundamentals of machine learning
- Implementation of gradient descent
- Data preprocessing and normalization
- Model evaluation techniques
- Data visualization best practices
