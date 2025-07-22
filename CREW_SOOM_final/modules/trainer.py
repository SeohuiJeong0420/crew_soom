import pandas as pd
import datetime
from datetime import timedelta

def preprocess_hourly_data():
    df = pd.read_csv("data/asos_seoul_hourly.csv")
    df['tm'] = pd.to_datetime(df['tm'])

    flood_periods = [
        ("2000-08-23", "2000-09-01"),
        ("2002-08-30", "2002-09-01"),
        ("2005-08-02", "2005-08-11"),
        ("2006-07-09", "2006-07-29"),
        ("2007-09-13", "2007-09-13"),
        ("2011-07-26", "2011-07-29"),
        ("2013-07-11", "2013-07-15"),
        ("2013-07-18", "2013-07-18"),
        ("2018-08-23", "2018-08-24"),
        ("2018-08-26", "2018-09-01"),
        ("2019-09-28", "2019-10-03"),
        ("2020-07-28", "2020-08-11"),
        ("2020-08-28", "2020-09-03"),
        ("2020-09-01", "2020-09-07"),
        ("2022-08-08", "2022-08-17"),
        ("2022-08-28", "2022-09-06")
    ]

    flood_dates = set()
    for start, end in flood_periods:
        date_range = pd.date_range(start=start, end=end, freq='H')
        flood_dates.update(date_range)

    df['flood_risk'] = df['tm'].isin(flood_dates).astype(int)
    df.to_csv("data/asos_seoul_hourly_with_flood_risk.csv", index=False)

def preprocess_daily_data():
    df = pd.read_csv("data/asos_seoul_daily.csv")
    df['tm'] = pd.to_datetime(df['tm'])
    df['month'] = df['tm'].dt.month
    df['dayofweek'] = df['tm'].dt.dayofweek
    df['year'] = df['tm'].dt.year
    df['sumRn'] = df['sumRn'].fillna(0)

    median_cols = ['minTa', 'maxTa', 'avgWs', 'avgTs', 'sumGsr', 'maxInsWs', 'sumSmlEv', 'avgPs']
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())

    df['ddMefs'] = df['ddMefs'].fillna(0)

    flood_periods = [
        ("2000-08-23", "2000-09-01"),
        ("2002-08-30", "2002-09-01"),
        ("2005-08-02", "2005-08-11"),
        ("2006-07-09", "2006-07-29"),
        ("2007-09-13", "2007-09-13"),
        ("2011-07-26", "2011-07-29"),
        ("2013-07-11", "2013-07-15"),
        ("2013-07-18", "2013-07-18"),
        ("2018-08-23", "2018-08-24"),
        ("2018-08-26", "2018-09-01"),
        ("2019-09-28", "2019-10-03"),
        ("2020-07-28", "2020-08-11"),
        ("2020-08-28", "2020-09-03"),
        ("2020-09-01", "2020-09-07"),
        ("2022-08-08", "2022-08-17"),
        ("2022-08-28", "2022-09-06")
    ]

    flood_dates = set()
    for start, end in flood_periods:
        start = datetime.datetime.strptime(start, "%Y-%m-%d")
        end = datetime.datetime.strptime(end, "%Y-%m-%d")
        while start <= end:
            flood_dates.add(start.date())
            start += timedelta(days=1)

    df['flood_risk'] = df['tm'].dt.date.isin(flood_dates).astype(int)
    df.loc[df['sumRn'] >= 30, 'flood_risk'] = 1

    df.to_csv("data/asos_seoul_daily_enriched.csv", index=False)

def preprocess_xgboost_features():
    import pandas as pd

    df = pd.read_csv("data/asos_seoul_daily_enriched.csv")
    df['tm'] = pd.to_datetime(df['tm'])

    df['month'] = df['tm'].dt.month
    df['day'] = df['tm'].dt.day
    df['weekday'] = df['tm'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    df['is_rainy'] = df['sumRn'].apply(lambda x: 1 if x >= 30 else 0)
    df['rain_hours'] = df['sumRn'].apply(lambda x: round(x / 3))
    df['max_hourly_rn'] = df['sumRn'].apply(lambda x: x if x <= 50 else 50)

    df.to_csv("data/asos_seoul_daily_enriched.csv", index=False)
    print("XGBoost용 파생 변수 추가 완료 및 저장됨.")