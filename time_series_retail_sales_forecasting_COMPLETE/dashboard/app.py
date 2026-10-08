from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/monthly_retail_sales.csv",parse_dates=["Date"])
df["MoM_Growth_%"]=df.Sales.pct_change()*100
df["YoY_Growth_%"]=df.Sales.pct_change(12)*100
df["3M_MA"]=df.Sales.rolling(3).mean()
df["12M_MA"]=df.Sales.rolling(12).mean()

st.set_page_config(page_title="Retail Sales Forecasting",page_icon="📈",layout="wide")
st.title("📈 Retail Sales Analytics & Forecasting")
st.caption("Interactive Time Series Analysis Dashboard")

years=sorted(df.Date.dt.year.unique())
selected=st.sidebar.multiselect("Select year(s)",years,default=years)
d=df[df.Date.dt.year.isin(selected)]

c1,c2,c3,c4=st.columns(4)
c1.metric("Total Sales",f"{d.Sales.sum():,.0f}")
c2.metric("Average",f"{d.Sales.mean():,.0f}")
c3.metric("Maximum",f"{d.Sales.max():,.0f}")
c4.metric("Minimum",f"{d.Sales.min():,.0f}")

st.subheader("Sales Trend")
st.plotly_chart(px.line(d,x="Date",y=["Sales","3M_MA","12M_MA"]),use_container_width=True)

a,b=st.columns(2)
with a:
    st.subheader("YoY Growth")
    st.plotly_chart(px.line(d,x="Date",y="YoY_Growth_%"),use_container_width=True)
with b:
    st.subheader("MoM Growth")
    st.plotly_chart(px.bar(d,x="Date",y="MoM_Growth_%"),use_container_width=True)

st.subheader("Monthly Seasonality")
season=d.assign(Month=d.Date.dt.month).groupby("Month",as_index=False).Sales.mean()
season["Month_Name"]=pd.to_datetime(season.Month,format="%m").dt.strftime("%b")
st.plotly_chart(px.bar(season,x="Month_Name",y="Sales"),use_container_width=True)

model_file=ROOT/"outputs/model_comparison.csv"
if model_file.exists():
    st.subheader("Model Comparison")
    models=pd.read_csv(model_file)
    st.dataframe(models,use_container_width=True)
    st.plotly_chart(px.bar(models,x="Model",y="RMSE"),use_container_width=True)

future=ROOT/"outputs/future_12_month_forecast.csv"
if future.exists():
    st.subheader("🔮 Next 12 Months")
    f=pd.read_csv(future,parse_dates=["Date"])
    st.plotly_chart(px.line(f,x="Date",y="Forecast_Sales",markers=True),use_container_width=True)
    st.dataframe(f,use_container_width=True)

ins=ROOT/"outputs/business_insights.txt"
if ins.exists():
    st.subheader("💡 Business Insights")
    st.code(ins.read_text())
