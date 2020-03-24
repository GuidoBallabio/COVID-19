from pathlib import Path

import numpy as np
import scipy
import pandas as pd

data_dir = Path(__file__).parent.parent / "data" 
national_data = data_dir / "dati-andamento-nazionale" / "dpc-covid19-ita-andamento-nazionale.csv"
regional_data = data_dir / "dati-regioni" / "dpc-covid19-ita-regioni.csv"
provincial_data = data_dir / "dati-province" / "dpc-covid19-ita-province.csv"

def extract_data(file, reg=False, prov=False):
    df = pd.read_csv(file, index_col=0, parse_dates=[0])

    df = df.drop(columns=["stato"])

    if reg or prov:
        df = df.drop(columns=["codice_regione", "lat", "long"])
        df = df.rename(columns={"denominazione_regione": "reg"})
    if prov:
        df = df.dropna()
        df = df.drop(columns=["denominazione_provincia", "codice_provincia"])
        df = df.rename(columns={"sigla_provincia": "prov"})

    df.index.name = "date"
    df.index = df.index.to_period(freq='D')
    
    if reg:
        df = df.reset_index().set_index(["reg", "date"]).sort_index()
    elif prov:
        df = df.reset_index().set_index(["reg", "prov", "date"]).sort_index()

    return df


def differentiate_column(df, column):
    series = df[column]
    return pd.DataFrame({column: series, 'diff1': series.diff(), 'diff2': series.diff().diff()})


def get_date_index(df):
    if isinstance(df.index, pd.MultiIndex):
        index = df.index.levels[-1]
    else:
        index = df.index
    return index

def get_date_range(index):
    if isinstance(index, pd.MultiIndex):
        return index.levels[-1].asi8 - index.levels[-1].asi8[0]
    return index.asi8 - index.asi8[0]


def add_days_since_start(n, index):
    return int(n) + index[0]


def ndays_since_start(date, df):
    if isinstance(df.index, pd.MultiIndex):
        return df.loc[date:,:].index.levels[-1].asi8 - df.index.levels[-1].asi8[0]
    return df.loc[date:,:].index.asi8[0] - df.index.asi8[0]


def summary_of_model(fitter, df):
    index = get_date_index(df)

    print(fitter)
    print("Peak: ", add_days_since_start(fitter.peak(), index))
    print("10% done:", add_days_since_start(fitter.inverse_perc(0.10), index))
    print("25% done:", add_days_since_start(fitter.inverse_perc(0.25), index))
    print("50% done:", add_days_since_start(fitter.inverse_perc(0.50), index))
    print("75% done:", add_days_since_start(fitter.inverse_perc(0.75), index))
    print("95% done:", add_days_since_start(fitter.inverse_perc(0.95), index))
    print("Plateau:", int(fitter.plateau()))


def summaries(models, df):
    for m in models:
        summary_of_model(m, df)


def collect_models(models, df, column, future):
    index = get_date_index(df)
    index = pd.period_range(index[0], index[-1]+future, name="date")
    mod_df = pd.DataFrame(df[column].reindex(index=index))
    ext_x = get_date_range(index)   

    for m in models:
        mod_df[m.__class__.__name__] = m.compute(ext_x)
    
    return mod_df
    