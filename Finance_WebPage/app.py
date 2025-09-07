import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred
import streamlit as st

st.title("📈 Buffett Indicator Dashboard")

# -----------------------------
# Caching functions
# -----------------------------
@st.cache_data(ttl=86400)  # cache for 1 day
def get_market_data():
    sp500 = yf.Ticker("^W5000")  # Wilshire 5000 = total US market
    data = sp500.history(period="max")
    data.index = data.index.tz_localize(None)
    return data

@st.cache_data(ttl=86400)
def get_gdp_data(api_key):
    fred = Fred(api_key=api_key)
    gdp = fred.get_series("GDP")
    gdp = gdp.resample("Q").mean()
    gdp = gdp.to_frame(name="GDP")  # convert Series to DataFrame with column name
    return gdp

# -----------------------------
# Fetch data
# -----------------------------
api_key = "02f2753d90abbf9cf8991396eaa0a290"  # Replace with your actual FRED API key
data = get_market_data()
gdp = get_gdp_data(api_key)

# -----------------------------
# Merge Market + GDP
# -----------------------------
combined = pd.merge(
    data[["Close"]],
    gdp,
    left_index=True,
    right_index=True,
    how="inner"
)

# Compute Buffett Indicator
combined["BuffettIndicator"] = combined["Close"] / combined["GDP"] * 100

# -----------------------------
# Polynomial Trend (2nd degree)
# -----------------------------
combined["Date_num"] = (combined.index - combined.index[0]).days
poly_line = np.polyfit(combined["Date_num"], combined["BuffettIndicator"], deg=2)
combined["Trend"] = np.polyval(poly_line, combined["Date_num"])

# Standard deviation of residuals
residuals = combined["BuffettIndicator"] - combined["Trend"]
std_dev = residuals.std()

# Standard deviation bands
combined["Trend_plus1SD"] = combined["Trend"] + std_dev
combined["Trend_plus2SD"] = combined["Trend"] + 2*std_dev
combined["Trend_minus1SD"] = combined["Trend"] - std_dev
combined["Trend_minus2SD"] = combined["Trend"] - 2*std_dev

# -----------------------------
# Show current Buffett Indicator
# -----------------------------
latest_ratio = combined["BuffettIndicator"].iloc[-1]
st.metric("Current Buffett Indicator", f"{latest_ratio:.2f}%")

# -----------------------------
# Plot
# -----------------------------
st.subheader("Buffett Indicator Over Time")
fig, ax = plt.subplots(figsize=(14,7))

ax.plot(combined.index, combined["BuffettIndicator"], color='blue', label="Buffett Indicator")
ax.plot(combined.index, combined["Trend"], color='gray', linestyle='--', label="Historical Trend")
ax.plot(combined.index, combined["Trend_plus1SD"], color='orange', linestyle='--', label="+1 SD")
ax.plot(combined.index, combined["Trend_plus2SD"], color='red', linestyle='--', label="+2 SD")
ax.plot(combined.index, combined["Trend_minus1SD"], color=(0.2, 0.7, 0.1), linestyle='--', label="-1 SD")
ax.plot(combined.index, combined["Trend_minus2SD"], color='green', linestyle='--', label="-2 SD")

ax.set_ylabel("Market Cap / GDP (%)")
ax.legend()
st.pyplot(fig)
