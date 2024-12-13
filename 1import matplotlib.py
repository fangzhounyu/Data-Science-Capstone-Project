import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random ESG data (emission reduction targets) and stock price changes
np.random.seed(42)
num_points = 100

# ESG metrics (emission reduction targets, in percentage)
esg_metrics = np.random.uniform(5, 40, num_points)  # ESG metrics range from 5% to 40%

# Stock price changes (simulating a relationship with ESG metrics, with some noise)
stock_price_changes = 0.5 * esg_metrics + np.random.normal(0, 5, num_points)  # Stock price increases with ESG, but with some noise

# Create the regression model
model = LinearRegression()
model.fit(esg_metrics.reshape(-1, 1), stock_price_changes)

# Get the regression line
regression_line = model.predict(esg_metrics.reshape(-1, 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(esg_metrics, stock_price_changes, color='skyblue', label='Data points', alpha=0.6)
plt.plot(esg_metrics, regression_line, color='red', linewidth=2, label='Regression Line')
plt.title("Scatter Plot with Regression Line: ESG Metrics vs. Stock Price Change", fontsize=14)
plt.xlabel("ESG Metrics (Emission Reduction Target %)")
plt.ylabel("Stock Price Change (%)")
plt.legend()
plt.grid(True)
plt.show()
