#!/usr/bin/env python
# coding: utf-8

# ### Python project for predicting stock volatility 

# In[1]:


pip install arch plotly pykalman


# In[2]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
from pykalman import KalmanFilter
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# In[3]:


# Load stock data for GOOGL

G_stock_df = yf.download('GOOGL', start="2011-01-01", end="2021-01-01")


# We calculate the log returns of GOOGL first in order to keep in mind the distribution factor

log_returns = np.log(G_stock_df['Adj Close']).diff().dropna()

log_returns


# Plotting the original log returns
plt.figure(figsize=(12, 6))
plt.plot(log_returns, label='Log Returns', color='blue')
plt.title("GOOGL Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.legend()
plt.show()


# In[4]:


# Method 1: Volatility Targeting
def volatility_targeting(log_returns, target_vol=0.15):
    actual_vol = log_returns.rolling(window=252).std() * np.sqrt(252)
    scaling_factor = target_vol / actual_vol
    return pd.Series(scaling_factor, index=log_returns.index)

# Calculating and plotting Volatility Targeting
scaling_factor_series = volatility_targeting(log_returns)
fig_vol_targeting = go.Figure()
fig_vol_targeting.add_trace(go.Scatter(x=scaling_factor_series.index, y=scaling_factor_series, mode='lines', name='Scaling Factor'))
fig_vol_targeting.update_layout(title="Volatility Targeting", xaxis_title="Date", yaxis_title="Scaling Factor")
fig_vol_targeting.show()


# In[5]:


# Method 2: Volatility Regime Switching
def volatility_regime_switching(log_returns, low_vol_threshold=0.15, high_vol_threshold=0.30):
    vol = log_returns.rolling(window=252).std() * np.sqrt(252)
    regime = pd.cut(vol, bins=[0, low_vol_threshold, high_vol_threshold, np.inf], labels=['low', 'medium', 'high'])
    return regime.reindex(G_stock_df.index, method='ffill')

# Calculating and plotting Volatility Regime Switching
volatility_regimes = volatility_regime_switching(log_returns)
fig = go.Figure()
for regime in ['low', 'medium', 'high']:
    regime_data = G_stock_df[volatility_regimes == regime]
    fig.add_trace(go.Scatter(
        x=regime_data.index,
        y=regime_data['Adj Close'],
        mode='markers',
        name=regime.capitalize() + ' Volatility',
        marker=dict(size=4),
        marker_symbol='circle',
        marker_line_width=1,
        marker_color={'low': 'green', 'medium': 'orange', 'high': 'red'}[regime]
    ))
fig.update_layout(
    title='GOOGL Stock Price with Volatility Regimes',
    xaxis_title='Date',
    yaxis_title='Adjusted Close Price',
    legend_title='Volatility Regime',
    template='plotly_dark'
)
fig.show()


# ### Trading strategies:

# 1. Mean Reversion Strategy
#  
# The Mean Reversion strategy is based on the principle that prices tend to revert back to their average or mean over time. This strategy utilizes Bollinger Bands, which are defined by a set number of standard deviations away from a moving average of the price.

# In[47]:


# Calculting the percentage gain/loss from a given trading strategy:
def calculate_percentage_gain_loss(strategy_returns):
    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) > 0:
        initial_value = strategy_returns.iloc[0]
        final_value = strategy_returns.iloc[-1]
        if initial_value != 0:
            return ((final_value - initial_value) / initial_value) * 100
        else:
            # If the initial value is zero, use the final value directly for percentage
            return final_value * 100
    else:
        return 0  # Return 0 if there are no valid returns
    
    
# Trading Strategy 1 - Mean Reversion strategy:
def mean_reversion_strategy(G_stock_df, window=20, num_std=2):
    if window > len(G_stock_df):
        return np.nan
    rolling_mean = G_stock_df['Adj Close'].rolling(window).mean()
    rolling_std = G_stock_df['Adj Close'].rolling(window).std()
    G_stock_df['Upper Band'] = rolling_mean + (rolling_std * num_std)
    G_stock_df['Lower Band'] = rolling_mean - (rolling_std * num_std)
    G_stock_df['Long Entry'] = G_stock_df['Adj Close'] < G_stock_df['Lower Band']
    G_stock_df['Long Exit'] = G_stock_df['Adj Close'] > rolling_mean
    G_stock_df['Positions Long'] = np.nan
    G_stock_df.loc[G_stock_df['Long Entry'], 'Positions Long'] = 1
    G_stock_df.loc[G_stock_df['Long Exit'], 'Positions Long'] = 0
    G_stock_df['Positions Long'] = G_stock_df['Positions Long'].fillna(method='pad')
    G_stock_df['Returns'] = G_stock_df['Adj Close'].pct_change()
    G_stock_df['Strategy Returns'] = G_stock_df['Returns'] * G_stock_df['Positions Long'].shift(1)
    strategy_returns = G_stock_df['Strategy Returns'].cumsum()
    return calculate_percentage_gain_loss(strategy_returns)


# 2. Momentum Strategy
#  
# The Momentum strategy capitalizes on the continuation of existing trends in the market. It uses moving averages to determine the trend direction and generates buy or sell signals accordingly.
# 
# 

# In[48]:


# Trading Strategy 2 - Momentum Strategy
def momentum_strategy(G_stock_df, short_window=40, long_window=100):
    if short_window > len(G_stock_df) or long_window > len(G_stock_df):
        return np.nan
    G_stock_df['Short MA'] = G_stock_df['Adj Close'].rolling(window=short_window, min_periods=1).mean()
    G_stock_df['Long MA'] = G_stock_df['Adj Close'].rolling(window=long_window, min_periods=1).mean()
    G_stock_df['Signal'] = np.where(G_stock_df['Short MA'] > G_stock_df['Long MA'], 1.0, 0.0)
    G_stock_df['Returns'] = G_stock_df['Adj Close'].pct_change()
    G_stock_df['Strategy Returns'] = G_stock_df['Returns'] * G_stock_df['Signal'].shift(1)
    strategy_returns = G_stock_df['Strategy Returns'].cumsum()
    return calculate_percentage_gain_loss(strategy_returns)


# 3. Pair Trading Strategy with Kalman Filter
#  
# Pair Trading with a Kalman Filter uses a statistical approach to identify mispricings in the market, allowing for a long position in undervalued assets and a short position in overvalued assets. The Kalman Filter is used to estimate the fair value dynamically.

# In[49]:


#  Trading Strategy 3 - Pair Trading Strategy with Kalman Filter
def pair_trading_strategy(G_stock_df):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=0, initial_state_covariance=1,
                      observation_covariance=1, transition_covariance=0.01)
    state_means, _ = kf.filter(G_stock_df['Adj Close'].values)
    G_stock_df['Kalman Filter'] = state_means.flatten()
    G_stock_df['Returns'] = G_stock_df['Adj Close'].pct_change()
    
    # Generating buy/sell signals with position limiting
    G_stock_df['Signal'] = np.where(G_stock_df['Adj Close'] < G_stock_df['Kalman Filter'], 1, -1)
    G_stock_df['Signal'] = np.clip(G_stock_df['Signal'], -1, 1)  # Limit position size

    G_stock_df['Strategy Returns'] = G_stock_df['Returns'] * G_stock_df['Signal'].shift(1)
    strategy_returns = G_stock_df['Strategy Returns'].cumsum()
    return calculate_percentage_gain_loss(strategy_returns)


# 4. Breakout Strategy
#  
# The Breakout Strategy aims to enter trades during periods of high market volatility, specifically when prices break out from their historical ranges.

# In[50]:


#  Trading Strategy 4 -  Breakout Strategy
def breakout_strategy(G_stock_df, window=50):
    if window > len(G_stock_df):
        return np.nan
    G_stock_df['Rolling Max'] = G_stock_df['Adj Close'].rolling(window).max()
    G_stock_df['Signal'] = np.where(G_stock_df['Adj Close'] > G_stock_df['Rolling Max'].shift(1), 1, 0)
    G_stock_df['Returns'] = G_stock_df['Adj Close'].pct_change()
    G_stock_df['Strategy Returns'] = G_stock_df['Returns'] * G_stock_df['Signal'].shift(1)
    strategy_returns = G_stock_df['Strategy Returns'].cumsum()
    return calculate_percentage_gain_loss(strategy_returns)


# In[51]:


# Load stock data
G_stock_df = yf.download('GOOGL', start="2000-01-01", end="2021-01-01")

# Apply strategies and calculate percentage gain/loss
mean_reversion_gain_loss = mean_reversion_strategy(G_stock_df.copy())
momentum_gain_loss = momentum_strategy(G_stock_df.copy())
pair_trading_gain_loss = pair_trading_strategy(G_stock_df.copy())
breakout_gain_loss = breakout_strategy(G_stock_df.copy())

# Print results
print("Mean Reversion Strategy Gain/Loss: {:.2f}%".format(mean_reversion_gain_loss))
print("Momentum Strategy Gain/Loss: {:.2f}%".format(momentum_gain_loss))
print("Pair Trading Strategy Gain/Loss: {:.2f}%".format(pair_trading_gain_loss))
print("Breakout Strategy Gain/Loss: {:.2f}%".format(breakout_gain_loss))


# In[52]:


# Creating the figure
fig = make_subplots(rows=1, cols=1)

# Adding scatter traces for each strategy's percentage gain/loss
fig.add_trace(go.Scatter(x=['Mean Reversion'], y=[mean_reversion_gain_loss], mode='markers+text', name='Mean Reversion Strategy', text=[f'{mean_reversion_gain_loss:.2f}%'], textposition='top center'))
fig.add_trace(go.Scatter(x=['Momentum'], y=[momentum_gain_loss], mode='markers+text', name='Momentum Strategy', text=[f'{momentum_gain_loss:.2f}%'], textposition='top center'))
fig.add_trace(go.Scatter(x=['Pair Trading'], y=[pair_trading_gain_loss], mode='markers+text', name='Pair Trading Strategy', text=[f'{pair_trading_gain_loss:.2f}%'], textposition='top center'))
fig.add_trace(go.Scatter(x=['Breakout'], y=[breakout_gain_loss], mode='markers+text', name='Breakout Strategy', text=[f'{breakout_gain_loss:.2f}%'], textposition='top center'))

# Updating layout
fig.update_layout(
    title='Strategy Percentage Gain/Loss Comparison',
    xaxis_title='Strategy',
    yaxis_title='Percentage Gain/Loss (%)',
    legend_title="Strategies",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True)
)

# Show plot
fig.show()


# In[ ]:





# In[ ]:




