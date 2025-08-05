# bitcoin_seasonality.py
# Proje 2 – Bitcoin’de Mevsimsellik ve Günlük Desen Analizi
# Visual Studio’da Python 3.x ortamında çalıştırın

# 1. Gerekli kütüphaneleri içe aktar
import pandas as pd                             # Veri işleme
import matplotlib.pyplot as plt                 # Grafik çizimi
import seaborn as sns                           # İstatistiksel görselleştirme
from pycoingecko import CoinGeckoAPI            # CoinGecko API’den veri çekme
from statsmodels.tsa.seasonal import seasonal_decompose  # Sezonel ayrıştırma
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # ACF/PACF

# 2. CoinGecko API ile veri çekme fonksiyonu
def get_btc_history(days=365):
    """
    CoinGecko’dan son `days` gününe ait Bitcoin fiyat verilerini çeker,
    saatlik verileri günlük ortalamaya dönüştürür ve döner.
    """
    cg = CoinGeckoAPI()  
    raw = cg.get_coin_market_chart_by_id(
        id='bitcoin',
        vs_currency='usd',
        days=days
    )
    # raw['prices'] → [[timestamp_ms, price], ...]
    df = pd.DataFrame(raw['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Günlük ortalama fiyat
    df = df.resample('D').mean()
    return df

# 3. Veriyi çek
btc = get_btc_history(365)

# 4. Özellik mühendisliği: haftanın günü ve ay adı sütunları ekle
btc['weekday'] = btc.index.day_name(locale='en_US')
btc['month']   = btc.index.month_name(locale='en_US')

# 5. Haftalık Desen – Boxplot
plt.figure(figsize=(10,5))
order_week = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.boxplot(
    x='weekday', y='price',
    data=btc,
    order=order_week
)
plt.title('BTC Fiyat Dağılımı – Hafta Günlerine Göre')
plt.xlabel('Hafta Günü')
plt.ylabel('Fiyat (USD)')
plt.tight_layout()
plt.savefig('btc_weekday_boxplot.png', dpi=150)
plt.show()

# 6. Aylık Desen – Boxplot
plt.figure(figsize=(12,6))
order_month = [
    'January','February','March','April','May','June','July',
    'August','September','October','November','December'
]
sns.boxplot(
    x='month', y='price',
    data=btc,
    order=order_month
)
plt.title('BTC Fiyat Dağılımı – Aylara Göre')
plt.xlabel('Ay')
plt.ylabel('Fiyat (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('btc_month_boxplot.png', dpi=150)
plt.show()

# 7. 30 Günlük Hareketli Ortalama
btc['MA30'] = btc['price'].rolling(window=30).mean()
plt.figure(figsize=(12,5))
plt.plot(btc.index, btc['price'], label='Fiyat')
plt.plot(btc.index, btc['MA30'], label='30 Günlük MA')
plt.title('Bitcoin Fiyatı ve 30 Günlük Hareketli Ortalama')
plt.xlabel('Tarih')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('btc_moving_average.png', dpi=150)
plt.show()

# 8. Sezonel Ayrıştırma (Trend + Seasonal + Resid)
decomp = seasonal_decompose(btc['price'], model='additive', period=30)
fig = decomp.plot()
fig.suptitle('Sezonel Ayrıştırma (Trend, Seasonal, Resid)')
plt.tight_layout()
fig.savefig('btc_seasonal_decompose.png', dpi=150)
plt.show()

# 9. ACF ve PACF – Otokorelasyon Analizi
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plot_acf(btc['price'].diff().dropna(), lags=30, ax=plt.gca())
plt.title('Otokorelasyon Fonksiyonu (ACF)')
plt.subplot(2,1,2)
plot_pacf(btc['price'].diff().dropna(), lags=30, ax=plt.gca())
plt.title('Kısmi Otokorelasyon Fonksiyonu (PACF)')
plt.tight_layout()
plt.savefig('btc_acf_pacf.png', dpi=150)
plt.show()
