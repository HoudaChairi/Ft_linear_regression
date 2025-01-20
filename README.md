# Ft_linear_regression 📈🚗

## 📝 Description
This project implements a simple linear regression model to predict car prices based on mileage. It serves as an introduction to machine learning fundamentals, focusing on:
- Implementation of gradient descent algorithm
- Understanding linear regression with a single feature
- Data visualization and model evaluation

## 🧠 Concepts Needed for the Project
* Linear Regression 📈
  - Understanding the relationship between independent and dependent variables
  - How to fit a line to data points
  - Predicting continuous values based on input features

* Gradient Descent 📉
  - Iterative optimization algorithm
  - Finding the minimum of the cost function
  - Updating parameters to improve predictions

* Loss Function 🧮
  - Measuring prediction errors
  - Mean Squared Error (MSE)
  - Cost function optimization

## 🧮 Mathematical Foundation
The price prediction is based on the following hypothesis:
```
estimatePrice(mileage) = θ₀ + (θ₁ * mileage)
```
Where:
- `θ₀` (theta0): Y-intercept
- `θ₁` (theta1): Slope of the line
- `mileage`: Input feature (X variable)

The model uses gradient descent to minimize the cost function with these update rules:
```
tmpθ₀ = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])
tmpθ₁ = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i]) * mileage[i]
```

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
