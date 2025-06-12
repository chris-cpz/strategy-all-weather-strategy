# all weather strategy
# Strategy Type: custom
# Description: all weather strategy
# Created: 2025-06-12T14:58:22.759Z

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the All Weather Strategy class
class AllWeatherStrategy:
    def __init__(self, data, risk_free_rate=0.01):
        # Initialize with market data and risk-free rate
        self.data = data
        self.risk_free_rate = risk_free_rate
        self.signals = pd.DataFrame(index=data.index)
        self.positions = pd.DataFrame(index=data.index)
        self.portfolio = pd.DataFrame(index=data.index)

    def generate_signals(self):
        # Generate trading signals based on moving averages
        self.signals['short_mavg'] = self.data['Close'].rolling(window=40, min_periods=1).mean()
        self.signals['long_mavg'] = self.data['Close'].rolling(window=100, min_periods=1).mean()
        self.signals['signal'] = 0
        self.signals['signal'][40:] = np.where(self.signals['short_mavg'][40:] > self.signals['long_mavg'][40:], 1, 0)
        self.signals['positions'] = self.signals['signal'].diff()

    def backtest_strategy(self):
        # Backtest the strategy
        self.positions['positions'] = self.signals['positions']
        self.portfolio['holdings'] = self.positions['positions'] * self.data['Close']
        self.portfolio['cash'] = 100000 - (self.positions['positions'] * self.data['Close']).cumsum()
        self.portfolio['total'] = self.portfolio['cash'] + self.portfolio['holdings']
        self.portfolio['returns'] = self.portfolio['total'].pct_change()

    def calculate_performance_metrics(self):
        # Calculate performance metrics
        returns = self.portfolio['returns']
        sharpe_ratio = (returns.mean() - self.risk_free_rate) / returns.std() * np.sqrt(252)
        drawdown = self.portfolio['total'].cummax() - self.portfolio['total']
        max_drawdown = drawdown.max()
        return sharpe_ratio, max_drawdown

    def plot_results(self):
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio['total'], label='Portfolio value')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()

# Generate sample data for demonstration
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='B')
prices = np.random.lognormal(mean=0.001, sigma=0.02, size=len(dates))
data = pd.DataFrame(data={'Close': prices}, index=dates)

# Main execution block
if __name__ == "__main__":
    # Instantiate the strategy with sample data
    strategy = AllWeatherStrategy(data)
    # Generate signals
    strategy.generate_signals()
    # Backtest the strategy
    strategy.backtest_strategy()
    # Calculate performance metrics
    sharpe_ratio, max_drawdown = strategy.calculate_performance_metrics()
    # Print performance metrics
    print("Sharpe Ratio:", sharpe_ratio)
    print("Max Drawdown:", max_drawdown)
    # Plot the results
    strategy.plot_results()

# Strategy Analysis and Performance
# Add your backtesting results and analysis here

# Risk Management
# Document your risk parameters and constraints

# Performance Metrics
# Track your strategy's key performance indicators:
# - Sharpe Ratio
# - Maximum Drawdown
# - Win Rate
# - Average Return
