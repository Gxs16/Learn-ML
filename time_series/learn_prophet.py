# -- coding: utf-8 --
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet
from prophet.plot import plot_components_plotly

# df = pd.read_excel("D:\\Work\\Study\\Learn-ML\\time_series\\data\\data.xlsx")
#
# df['datetime'] = pd.to_datetime(df['date_id'] + ' ' + df['start_time'])
#
# df.drop_duplicates(subset=['datetime'], inplace=True)
# df.drop(columns=['date_id', 'start_time'], inplace=True)
# df.to_csv("D:\\Work\\Study\\Learn-ML\\time_series\\data\\data.csv", index=False)

df = pd.read_csv("D:\\Work\\Study\\Learn-ML\\time_series\\data\\data.csv")
df['ds'] = pd.to_datetime(df['ds'])
df = df[df['ds'] >= pd.to_datetime(pd.to_datetime('2022-01-01'))]


m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
plot_components_plotly(m, forecast)

# fig2 = m.plot_components(forecast)
# plt.show()
print(1)