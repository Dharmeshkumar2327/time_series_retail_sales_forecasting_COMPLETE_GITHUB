from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
BASE=Path(__file__).resolve().parent
DATA=BASE/"data"/"monthly_retail_sales.csv"
OUT=BASE/"outputs"; OUT.mkdir(exist_ok=True)

def mape(y,p):
    y,p=np.asarray(y),np.asarray(p)
    mask=y!=0
    return np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100

def save(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=160,bbox_inches="tight"); plt.close()

df=pd.read_csv(DATA,parse_dates=["Date"]).sort_values("Date").set_index("Date")
sales=df["Sales"].asfreq("MS")

df["Month"]=df.index.month
df["Year"]=df.index.year
df["Quarter"]=df.index.quarter
df["MoM_Growth_%"]=df.Sales.pct_change()*100
df["YoY_Growth_%"]=df.Sales.pct_change(12)*100
df["3M_MA"]=df.Sales.rolling(3).mean()
df["6M_MA"]=df.Sales.rolling(6).mean()
df["12M_MA"]=df.Sales.rolling(12).mean()

plt.figure(figsize=(13,5)); plt.plot(sales); plt.title("Monthly Retail Sales"); plt.grid(alpha=.25); save("01_sales_trend.png")

plt.figure(figsize=(13,5)); plt.plot(sales,label="Sales",alpha=.5); plt.plot(df["3M_MA"],label="3M MA"); plt.plot(df["12M_MA"],label="12M MA"); plt.legend(); plt.title("Rolling Averages"); plt.grid(alpha=.25); save("02_rolling_averages.png")

monthly=df.groupby(df.index.month).Sales.mean()
plt.figure(figsize=(10,5)); plt.bar(range(1,13),monthly); plt.xticks(range(1,13),["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]); plt.title("Average Sales by Month"); plt.grid(axis="y",alpha=.25); save("03_seasonality.png")

plt.figure(figsize=(13,5)); plt.plot(df.index,df["YoY_Growth_%"]); plt.axhline(0,ls="--"); plt.title("YoY Growth"); plt.grid(alpha=.25); save("04_yoy_growth.png")

q1,q3=sales.quantile([.25,.75]); iqr=q3-q1
df["Is_Outlier"]=(sales<q1-1.5*iqr)|(sales>q3+1.5*iqr)
plt.figure(figsize=(13,5)); plt.plot(sales); plt.scatter(df.index[df.Is_Outlier],df.loc[df.Is_Outlier,"Sales"],s=50); plt.title("Anomaly Detection"); plt.grid(alpha=.25); save("05_anomalies.png")

dec=seasonal_decompose(sales,model="additive",period=12)
fig=dec.plot(); fig.set_size_inches(13,9); fig.savefig(OUT/"06_decomposition.png",dpi=160); plt.close(fig)

adf=adfuller(sales.dropna())
pd.DataFrame({"Metric":["ADF Statistic","p-value"],"Value":[adf[0],adf[1]]}).to_csv(OUT/"adf_test.csv",index=False)

fig,ax=plt.subplots(figsize=(12,5)); plot_acf(sales.dropna(),ax=ax,lags=30); fig.savefig(OUT/"07_acf.png",dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5)); plot_pacf(sales.dropna(),ax=ax,lags=30,method="ywm"); fig.savefig(OUT/"08_pacf.png",dpi=160); plt.close(fig)

train,test=sales.iloc[:-12],sales.iloc[-12:]
results=[]; forecasts={}

hw=ExponentialSmoothing(train,trend="add",seasonal="add",seasonal_periods=12).fit()
p=hw.forecast(12); results.append(["Holt-Winters",mean_absolute_error(test,p),np.sqrt(mean_squared_error(test,p)),mape(test,p)]); forecasts["Holt-Winters"]=p

sar=SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,12),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
p=sar.forecast(12); results.append(["SARIMA",mean_absolute_error(test,p),np.sqrt(mean_squared_error(test,p)),mape(test,p)]); forecasts["SARIMA"]=p

ml=pd.DataFrame({"Sales":sales})
for lag in [1,2,3,6,12]: ml[f"Lag_{lag}"]=ml.Sales.shift(lag)
ml["Roll3"]=ml.Sales.shift(1).rolling(3).mean(); ml["Roll6"]=ml.Sales.shift(1).rolling(6).mean()
ml["Month"]=ml.index.month; ml["Quarter"]=ml.index.quarter; ml=ml.dropna()
features=["Lag_1","Lag_2","Lag_3","Lag_6","Lag_12","Roll3","Roll6","Month","Quarter"]
tr=ml[ml.index<test.index[0]]; te=ml[ml.index>=test.index[0]]
rf=RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=2,random_state=42).fit(tr[features],tr.Sales)
p=pd.Series(rf.predict(te[features]),index=te.index)
results.append(["Random Forest",mean_absolute_error(te.Sales,p),np.sqrt(mean_squared_error(te.Sales,p)),mape(te.Sales,p)]); forecasts["Random Forest"]=p

comparison=pd.DataFrame(results,columns=["Model","MAE","RMSE","MAPE"]).sort_values("RMSE")
comparison.to_csv(OUT/"model_comparison.csv",index=False)
plt.figure(figsize=(9,5)); plt.bar(comparison.Model,comparison.RMSE); plt.xticks(rotation=20); plt.title("Model Comparison - RMSE"); plt.grid(axis="y",alpha=.25); save("09_model_comparison.png")

plt.figure(figsize=(13,6)); plt.plot(test,label="Actual",lw=3)
for name,p in forecasts.items(): plt.plot(p,ls="--",label=name)
plt.legend(); plt.title("Forecast Comparison"); plt.grid(alpha=.25); save("10_forecast_comparison.png")

best=comparison.iloc[0].Model
if best=="Holt-Winters":
    final=ExponentialSmoothing(sales,trend="add",seasonal="add",seasonal_periods=12).fit(); future=final.forecast(12)
elif best=="SARIMA":
    final=SARIMAX(sales,order=(1,1,1),seasonal_order=(1,1,1,12),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False); future=final.forecast(12)
else:
    hist=sales.copy(); vals=[]; idx=pd.date_range(sales.index[-1]+pd.offsets.MonthBegin(1),periods=12,freq="MS")
    for d in idx:
        row=pd.DataFrame([{"Lag_1":hist.iloc[-1],"Lag_2":hist.iloc[-2],"Lag_3":hist.iloc[-3],"Lag_6":hist.iloc[-6],"Lag_12":hist.iloc[-12],"Roll3":hist.iloc[-3:].mean(),"Roll6":hist.iloc[-6:].mean(),"Month":d.month,"Quarter":d.quarter}])
        pred=rf.predict(row[features])[0]; vals.append(pred); hist.loc[d]=pred
    future=pd.Series(vals,index=idx)

future_df=pd.DataFrame({"Date":future.index,"Forecast_Sales":future.values})
future_df.to_csv(OUT/"future_12_month_forecast.csv",index=False)
plt.figure(figsize=(13,6)); plt.plot(sales,label="Historical"); plt.plot(future,label=f"Forecast - {best}",ls="--",lw=3); plt.legend(); plt.title("Next 12 Months Forecast"); plt.grid(alpha=.25); save("11_future_forecast.png")

df.to_csv(OUT/"engineered_sales_data.csv")
ins=f"""Best model: {best}
Best month: {pd.Timestamp(2024,monthly.idxmax(),1).strftime("%B")}
Lowest month: {pd.Timestamp(2024,monthly.idxmin(),1).strftime("%B")}
Detected outliers: {int(df.Is_Outlier.sum())}
Next 12-month forecast total: {future.sum():,.2f}
Average monthly forecast: {future.mean():,.2f}
"""
(OUT/"business_insights.txt").write_text(ins)
print(ins)
print("Analysis completed successfully.")
