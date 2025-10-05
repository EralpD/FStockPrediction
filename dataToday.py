import yfinance as yf
import datetime

def getData(ticker:str, day_intv:str, startD):
    today = str(datetime.datetime.now()).split(" ")[0]
    data = yf.download(ticker, start=startD, end=today, interval=day_intv)

    return data, today