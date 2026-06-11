# ------------------------------------
# LINEAR REGRESSION
# ------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression

# Ad spend is the input variable.  Values are wrapped in [] to create columns for processing.
X = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)

# Revenue is the output variable.
y = np.array([150, 290, 440, 600, 720])

# Establishing my empty calculator aka the linear regression model
model = LinearRegression()

# Training the model to study the relationship between ad spend and revenue
# Model performs required caculations, and stores outputs for later retrieval
model.fit(X, y)

# Linear regression learns two numbers; slope and intercept
# Slope = how much revenue changes when ad spend increase by 1
# Intercept = predicted revenue when ad spend is 0
slope = model.coef_[0]
intercept = model.intercept_

# Print the learned linear regression equation
# Shortened intecept and slope to 2 decimal places for readabiity in the regression equation
print("Slope:", slope)
print("Intercept:", intercept)
print(f"Regression equation: y = {intercept:.2f} + {slope:.2f}x")

# The predict method plugs X into the learned regression equation to calculate predicted revenue
predictions = model.predict(X)

# Print revenue predictions for given data
print("\nPredicted revenue values:")
for ad_spend, predicted_revenue in zip(X.flatten(), predictions):
    print(f"Ad Spend = {ad_spend}, Predicted Revenue = {predicted_revenue:.2f}")

# Predict revenue for ad spend of $60K
new_ad_spend = np.array([[60]])
predicted_new_revenue = model.predict(new_ad_spend)

print(f"\nPredicted revenue for ad spend of 60: {predicted_new_revenue[0]:.2f}")


# ------------------------------------
# PLOTTING
# ------------------------------------
import matplotlib.pyplot as plot

plot.scatter(X, y, label="Actual Data")     # Adds given data points to the plot
plot.plot(X, predictions, label="Regression Line")     # Adds the fitted regression line to the plot
plot.xlabel("Ad Spend ($ in 1000s)")     # X axis label
plot.ylabel("Revenue ($ in 1000s)")      # Y axis label
plot.title("Linear Regression: Ad Spend vs Revenue")     # Plot label
plot.legend()     # Add plot legend
plot.show()     # Display the plot


# ------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------

# The numpy library enables working with numbers in arrays
import numpy as np

# The problem requires binary decision-making so LogisticRegression is the appropriate ML model
from sklearn.linear_model import LogisticRegression


# Weekly usage hours is the input variable.  Values are wrapped in [] to create columns for processing.
X = np.array([[1], [3], [5], [7], [9]])

# Renewal status is the output variable to predict. 0 means did not renew.  1 means did renew.
y = np.array([0, 0, 1, 1, 1])


# Establishing my empty calculator aka the logistic regression model
model = LogisticRegression()


# Training the model to study the relationship between usage hours and renewal
# Model performs required calculations, and stores outputs for later retrieval
model.fit(X, y)


# Logistic regression learns two numbers; beta0 - start of the curve, and beta1 - how much p changes when x increases
# Extracting both values from the model to plug them into the equation later
b0 = model.intercept_[0]
b1 = model.coef_[0][0]


# Print the learned logistic regression equation
# Shortened both Beta values to 4 decimal places for readability
print("Logistic Regression Equation:")
print(f"p = 1 / (1 + e^(-({b0:.4f} + {b1:.4f} * x)))")



# ------------------------------------
# MODEL SELECTION
# ------------------------------------
import math

# given values
beta_0 = -2
beta_1 = 0.8
x = 5

# the logistic regression formula
# p = 1 / (1 + e ^ -(β0 + β1x))
# Calculating inside the parentheses first and assigning it to "z" for easy plug and play
z = beta_0 + beta_1 * x      #Output: 2.0

# Plugging in z for full logistic regression calculation
# Euler's number is accounted for in "math.exp", making this 1/(1 + e^(-2.0))
p = 1 / (1 + math.exp(-z))     

# printing the probability of disease diagnosis with a test score of 5
print("z =", z)
print("The probability of disease diagnosis, given a test score of 5 =", p)



