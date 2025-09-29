import yfinance as yf
import datetime

def getData(ticker:str, day_intv:str):
    today = str(datetime.datetime.now()).split(" ")[0]
    data = yf.download(ticker, start="2020-01-01", end=today, interval=day_intv)

    return data, today