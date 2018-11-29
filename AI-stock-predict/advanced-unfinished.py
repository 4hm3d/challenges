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

class MyLocator(mticker.MaxNLocator):
    def __init__(self, *args, **kwargs):
        mticker.MaxNLocator.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return mticker.MaxNLocator.__call__(self, *args, **kwargs)


mondays = YearLocator(1)        # major ticks on the mondays
alldays = WeekdayLocator(byweekday=MO, interval=1)              # minor ticks on the days
weekFormatter = DateFormatter('%m/%d/%y')  # e.g., Jan 12
dayFormatter = DateFormatter('%I:%M%p')      # e.g., 12


file_name = "SIX.csv"
raw_data = open(file_name, 'r').read().strip().replace("\r", "").split("\n")

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


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else: #Over midnight
        return nowTime >= startTime or nowTime <= endTime


cal = calendar()
holidays = cal.holidays(
    start=raw_data[0].split(',')[0],
    end=raw_data[len(raw_data)-1].split(',')[0]
)

X = []
Y = []
i = 0
year = 0
#Date,Open,High,Low,Close,Adj Close,Volume
for m in range(1, len(raw_data)):
    last_minute_row = raw_data[ i-1 ].split(",")
    row = raw_data[i].split(",")

    last_minute_row[0] += "AM" if int(last_minute_row[0].split(' ')[1][0:1]) < 12 else "PM"
    row[0] += "AM" if int(row[0].split(' ')[1][0:1]) < 12 else "PM"
    
    is_holiday = 1 if pd.to_datetime([row[0].split(" ")[0]]).isin(holidays)[0] == True else 0
    season = get_season(row[0].split(' ')[0])
    month = int( row[0].split('-')[1] )

    last_minute_row = [
        datetime.strptime(last_minute_row[0], "%Y-%m-%d %H:%M%p"),
        int(last_minute_row[1]),
        float(last_minute_row[2]), #open
        float(last_minute_row[3]), #high
        float(last_minute_row[4]), #low
        float(last_minute_row[5]), #close
        int(last_minute_row[6])
    ]
    row = [
        datetime.strptime(row[0], "%Y-%m-%d %H:%M%p"),
        float(row[1]),
        float(row[2]), #open
        float(row[3]), #high
        float(row[4]), #low
        float(row[5]), #close
        int(row[6])
    ]
    
    X.append([
        int(row[0].year),
        int(row[0].month),
        int(row[0].day),
        int(row[0].hour),
        last_minute_row[0].timestamp(),
        last_minute_row[2],
        last_minute_row[3],
        last_minute_row[4],
        last_minute_row[5],
        last_minute_row[6],
        is_holiday,
        season
    ])

    Y.append([
        row[2], #open
        row[3], #high
        row[4], #low
        row[5], #close
        row[6]
    ])
    i += 1
#for 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0, random_state=None, shuffle=False)

from sklearn.neural_network import MLPRegressor
my_classifier = MLPRegressor(solver ="lbfgs", activation="logistic", shuffle=False)

my_classifier.fit(X_train, Y_train)



"""

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')
plt.show()


sys.exit(0)
"""

last_row = raw_data[len(raw_data)-1].split(",")
last_row[0] += "AM" if int(last_row[0].split(' ')[1][0:1]) < 12 else "PM"
last_row[0] = datetime.strptime(last_row[0], "%Y-%m-%d %H:%M%p")
last_row = [
    last_row[0],
    float(last_row[2]), #open
    float(last_row[3]), #high
    float(last_row[4]), #low
    float(last_row[5]), #close
    int(last_row[6])
]


new_X = []

results = []
for date in daterange(last_row[0].date(), last_row[0].replace(year=last_row[0].year+1)):
    #market hours only
    timeStart = date = date.replace(hour=9)
    timeEnd =  date.replace(hour=16)
    
    for i in range(1,12):
        date = date.replace(hour=date.hour+1)
        if isNowInTimePeriod( timeStart, timeEnd, date  ):
            current_date = str(date.year)+"-"+str(date.month)+"-"+str(date.day)
            is_holiday = 1 if pd.to_datetime([ current_date ]).isin(holidays)[0] == True else 0
            season = get_season( current_date )
            
            new_X = [ [
                int(date.year),
                int(date.month),
                int(date.day),
                int(date.hour),
                last_row[1],
                last_row[2],
                last_row[3],
                last_row[4],
                last_row[5],
                date.month,
                is_holiday,
                season
            ] ]

            predictions = my_classifier.predict(new_X)
            
            last_row = [
                date,
                float(predictions[0][0]), #open
                float(predictions[0][1]), #high
                float(predictions[0][2]), #low
                float(predictions[0][3]), #close
                int(predictions[0][4])
            ]
            results.append([
                date,
                float(predictions[0][0]), #open
                float(predictions[0][1]), #high
                float(predictions[0][2]), #low
                float(predictions[0][3]), #close
                int(predictions[0][4])
            ])
        else:
            continue
        


#draw trained

print('#1')

X_final = [[datetime(item[0], item[1], item[2], item[3])] for item in X_train] + [[item[0]] for item in results]
Y_final = Y_train + [[item[1],item[2],item[3],item[4]] for item in results]

#X_final = [[datetime(item[0], item[1], item[2], item[3])] for item in X_test]
#Y_final = Y_test
df = pd.DataFrame({
        'date':mdates.date2num( [  item[0] for item in X_final] ),
        'open':[item[0] for item in Y_final],
        'high':[item[1] for item in Y_final],
        'low':[item[2] for item in Y_final],
        'close':[item[3] for item in Y_final],
})

print(results)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)

ax.yaxis.set_major_locator(MyLocator(5, prune='both'))

candlestick_ohlc(
    ax,
    zip(
        df['date'],
        df['open'],
        df['high'],
        df['low'],
        df['close']
    ),
    width=0.108,
    colorup='g',
    colordown='r',
    alpha=0.0
)


print('#4')
plt.title('SIX Flags AI prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')
plt.show()