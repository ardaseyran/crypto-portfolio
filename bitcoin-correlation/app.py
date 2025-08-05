import streamlit as st
import pandas as pd
from pycoingecko import CoinGeckoAPI

st.set_page_config(page_title="BTC Korelasyon", layout="wide")
st.title("Bitcoin–Altcoin Korelasyon Analizi")

cg = CoinGeckoAPI()
@st.cache_data
def load_data(days=90):
    data = cg.get_coin_market_chart_by_id('bitcoin','usd',days)
    prices = pd.DataFrame(data['prices'], columns=['ts','btc'])
    prices.ts = pd.to_datetime(prices.ts, unit='ms')
    prices.set_index('ts', inplace=True)
    # Diğer coin’leri de ekleyelim
    coins = ['ethereum','ripple','solana','cardano','binancecoin']
    for c in coins:
        d = cg.get_coin_market_chart_by_id(c,'usd',days)
        df = pd.DataFrame(d['prices'], columns=['ts',c])
        df.ts = pd.to_datetime(df.ts, unit='ms')
        df.set_index('ts', inplace=True)
        prices = prices.join(df, how='outer')
    return prices.resample('D').mean().dropna()

df = load_data()

st.subheader("Isı Haritası (Pearson Korelasyon)")
st.dataframe(df.corr())

st.subheader("Fiyat Zaman Serisi")
st.line_chart(df)

# Rolling korelasyon
st.subheader("30 Günlük Rolling Korelasyon (BTC vs Diğerleri)")
rolling = df.rolling(30).corr(df['btc'])
for coin in df.columns.drop('btc'):
    st.line_chart(rolling[coin])
