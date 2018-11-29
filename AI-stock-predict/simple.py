#pip install -U scikit-learn

import time, sys, sklearn
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from datetime import date, datetime, timedelta
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, DateFormatter, WeekdayLocator, YearLocator
import matplotlib.ticker as mticker
from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR

def daterange(start_date, end_date):
  return rrule(DAILY, dtstart=start_date, until=end_date, byweekday=(MO,TU,WE,TH,FR))

mondays = YearLocator(1)        # major ticks on the mondays
alldays = WeekdayLocator(byweekday=MO, interval=1)              # minor ticks on the days
weekFormatter = DateFormatter('%m/%d/%y')  # e.g., Jan 12
dayFormatter = DateFormatter('%I:%M%p')      # e.g., 12


file_name = "MSFT.csv"

"""
winter = 1
spring = 2
summer = 3
autumn = 4
"""
DummyYear = 2000 # StockOverFlow :) im laxy
seasons = [(1, (date(DummyYear,  1,  1),  date(DummyYear,  3, 20))),
           (2, (date(DummyYear,  3, 21),  date(DummyYear,  6, 20))),
           (3, (date(DummyYear,  6, 21),  date(DummyYear,  9, 22))),
           (4, (date(DummyYear,  9, 23),  date(DummyYear, 12, 20))),
           (1, (date(DummyYear, 12, 21),  date(DummyYear, 12, 31)))]

def get_season(now):
    now = datetime.strptime(now, "%Y-%m-%d")
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=DummyYear)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


cal = calendar()


X = []
Y = []
i = 0
year = 0
#Date,Open,High,Low,Close,Adj Close,Volume

raw_data = pd.read_csv(file_name,
                     parse_dates=[0],
                     infer_datetime_format=True)


holidays = cal.holidays(
    start=raw_data.iloc[0].date,
    end=raw_data.iloc[-1].date
)
i = 0
for index, row in raw_data.iterrows():

    is_holiday = 1 if pd.to_datetime([row.date]).isin(holidays)[0] == True else 0
    season = get_season( str(row.date.date()) )
    month = int( row.date.month )

    X.append([
        i/(360*9),
        int(row.date.month),
        is_holiday,
        season
    ])

    Y.append([
        #row.open, #open
        #row.high, #high
        row.low, #low
        #row.close, #close
        #row.volume #volume
    ])
    i += 1


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0)

from sklearn.neural_network import MLPRegressor
my_classifier = MLPRegressor(solver ="lbfgs", activation="logistic", shuffle=False)

my_classifier.fit(X_train, Y_train)




new_X = []

results = []
tmp_row = raw_data.iloc[-1]
for date in daterange(
    row.date.replace(day=row.date.day+1),
    row.date.replace(year=row.date.year+2)
):
    tmp_row.date = date
    is_holiday = 1 if pd.to_datetime([tmp_row.date]).isin(holidays)[0] == True else 0
    season = get_season( str(tmp_row.date.date()) )
    month = int( tmp_row.date.month )
    
    new_X = [ [
        i/(360*9),
        int(tmp_row.date.month),
        is_holiday,
        season
    ] ]


    predictions = my_classifier.predict(new_X)

    tmp_pd = pd.DataFrame({
        'date':[tmp_row.date],
        'open':[float(predictions[0])],
        'high':[float(predictions[0])],
        'low':[float(predictions[0])],
        'close':[float(predictions[0])],
        'volume':[float(predictions[0])],
        #'high':[float(predictions[0][1])],
        #'low':[float(predictions[0][2])],
        #'close':[float(predictions[0][3])],
        #'volume':[int(predictions[0][4])],
    })
    raw_data = [raw_data, tmp_pd]
    raw_data = pd.concat(raw_data)
    i += 1


print(raw_data)
#draw trained

print('#1')


fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)


candlestick_ohlc(
    ax,
    zip(
        mdates.date2num(raw_data.date),
        raw_data['open'],
        raw_data['high'],
        raw_data['low'],
        raw_data['close']
    ),
    width=0.108,
    colorup='g',
    colordown='r',
    alpha=0.0
)
plt.plot(mdates.date2num(raw_data.date),raw_data['low'])


print('#4')
plt.title('MSFT AI prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')
plt.show()