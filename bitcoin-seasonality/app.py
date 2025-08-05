import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(page_title="BTC Sezon & Desen", layout="wide")
st.title("Bitcoin Mevsimsellik & Günlük Desen Analizi")

# Veri çekme
cg = CoinGeckoAPI()
@st.cache_data
def get_btc(days=365):
    data = cg.get_coin_market_chart_by_id('bitcoin','usd',days)
    df = pd.DataFrame(data['prices'], columns=['ts','price'])
    df.ts = pd.to_datetime(df.ts, unit='ms')
    df.set_index('ts', inplace=True)
    return df.resample('D').mean().dropna()
df = get_btc()

# Haftanın günü boxplot
st.subheader("Haftanın Günlerine Göre Dağılım")
df['weekday'] = df.index.day_name()
fig1, ax1 = plt.subplots()
df.boxplot(column='price', by='weekday', ax=ax1, grid=False)
ax1.set_xlabel(""); ax1.set_ylabel("Fiyat (USD)")
st.pyplot(fig1)

# Aylara göre boxplot
st.subheader("Aylara Göre Dağılım")
df['month'] = df.index.month_name()
fig2, ax2 = plt.subplots()
df.boxplot(column='price', by='month', ax=ax2, grid=False)
ax2.set_xlabel(""); ax2.set_ylabel("Fiyat (USD)")
st.pyplot(fig2)

# 30-Günlük MA
st.subheader("30-Günlük Hareketli Ortalama")
df['MA30'] = df['price'].rolling(30).mean()
fig3, ax3 = plt.subplots()
ax3.plot(df.index, df['price'], label='Günlük Fiyat')
ax3.plot(df.index, df['MA30'], label='MA30')
ax3.legend(); ax3.set_ylabel("Fiyat (USD)")
st.pyplot(fig3)

# ACF & PACF
st.subheader("ACF & PACF")
fig4, ax4 = plt.subplots(2,1, figsize=(8,6))
plot_acf(df['price'], lags=30, ax=ax4[0])
plot_pacf(df['price'], lags=30, ax=ax4[1])
st.pyplot(fig4)
