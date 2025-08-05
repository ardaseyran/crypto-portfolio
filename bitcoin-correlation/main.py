import pandas as pd
from pycoingecko import CoinGeckoAPI
import matplotlib.pyplot as plt
from datetime import datetime
import time

# CoinGecko API nesnesi
cg = CoinGeckoAPI()

# Çekilecek coin listesi
coins = ['bitcoin', 'ethereum', 'ripple', 'solana', 'cardano', 'binancecoin']

# Coin fiyatlarını çekmek için fonksiyon
def get_coin_history(coin_id, days=90):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        prices = data['prices']  # [timestamp, price] listesi
        print(f"{coin_id} için {len(prices)} kayıt getirildi.")
        # DataFrame'e dönüştür
        df = pd.DataFrame(prices, columns=['timestamp', coin_id])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return pd.DataFrame()

# Tüm coin verilerini tek DataFrame'de birleştir
df = pd.DataFrame()
for coin in coins:
    prices = get_coin_history(coin)
    if df.empty:
        df = prices
    else:
        df = df.join(prices, how='outer')
    time.sleep(1)  # API'yı zorlamamak için bekle

# Günlük kapanış fiyatı hesapla
df = df.resample('D').mean()

# DataFrame hakkında bilgi yazdır
print("DataFrame info:")
print(df.info())
print("Her sütundaki NaN sayısı:")
print(df.isna().sum())

# Korelasyon matrisi (Pearson)
corr_matrix = df.corr()
print("Korelasyon Matrisi (Pearson):")
print(corr_matrix)

# --- Spearman korelasyonu hesapla ---
# Spearman korelasyonu, sıralı ilişkiyi gösterir ve outlier'lara karşı daha dayanıklıdır.
corr_spearman = df.corr(method='spearman')
print("\nSpearman Korelasyon Matrisi:")
print(corr_spearman)

# Grafik: fiyatların zaman serisi
plt.figure(figsize=(14, 6))
for coin in coins:
    plt.plot(df.index, df[coin], label=coin)
plt.title("Kripto Para Fiyatları (Son 90 Gün)")
plt.xlabel("Tarih")
plt.ylabel("Fiyat (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("fiyat_grafik.png", dpi=150)
plt.show()

# Grafik: Korelasyon ısı haritası (Pearson)
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(coins)), coins, rotation=45)
plt.yticks(range(len(coins)), coins)
plt.title("Korelasyon Matrisi (Pearson)")
plt.tight_layout()
plt.savefig("korelasyon_grafik.png", dpi=150)
plt.show()

# Grafik: Korelasyon ısı haritası (Spearman)
plt.figure(figsize=(8, 6))
plt.imshow(corr_spearman, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(coins)), coins, rotation=45)
plt.yticks(range(len(coins)), coins)
plt.title("Korelasyon Matrisi (Spearman)")
plt.tight_layout()
plt.savefig("korelasyon_spearman_grafik.png", dpi=150)
plt.show()

# --- Rolling Korelasyon ---
# Rolling korelasyon, zaman içinde korelasyonun değişip değişmediğini gösterir.
window_size = 30  # 30 günlük pencere
rolling_corr = df.rolling(window=window_size).corr(df['bitcoin'])

plt.figure(figsize=(12,6))
for coin in coins:
    if coin != 'bitcoin':
        plt.plot(rolling_corr.index, rolling_corr[coin], label=coin)
plt.title(f"BTC ile {window_size} Günlük Rolling Korelasyon")
plt.xlabel("Tarih")
plt.ylabel("Korelasyon")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("rolling_korelasyon.png", dpi=150)
plt.show()

# --- Lag/Lead Korelasyon ---
# Lag/Lead korelasyon, BTC'nin diğer coinlerden önce mi sonra mı hareket ettiğini (etkilediğini) gösterir.
def lag_corr(base, target, max_lag=7):
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = base.corr(target.shift(-lag))
        else:
            corr = base.shift(lag).corr(target)
        out[lag] = corr
    return pd.Series(out)

lags = {}
for coin in coins:
    if coin != 'bitcoin':
        lags[coin] = lag_corr(df['bitcoin'], df[coin], max_lag=7)
lags_df = pd.DataFrame(lags)

print("\nBTC ile gecikmeli korelasyonlar (satırlar lag günü):")
print(lags_df)

plt.figure(figsize=(10,5))
for coin in lags_df.columns:
    plt.plot(lags_df.index, lags_df[coin], label=coin)
plt.axvline(0, linestyle='--', color='black')
plt.title("BTC ile Lag/Lead Korelasyonlar (±7 gün)")
plt.xlabel("Lag (gün)")
plt.ylabel("Korelasyon")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("lag_lead_korelasyon.png", dpi=150)
plt.show()
print("Analiz tamamlandı.")
