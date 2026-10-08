from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_dataset():
    df=pd.read_csv(ROOT/'data/monthly_retail_sales.csv')
    assert list(df.columns)==['Date','Sales']
    assert len(df)>=24
    assert df.Sales.notna().all()

def test_files():
    assert (ROOT/'main.py').exists()
    assert (ROOT/'dashboard/app.py').exists()
    assert (ROOT/'README.md').exists()
