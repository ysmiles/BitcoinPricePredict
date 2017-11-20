from exchanges.coindesk import CoinDesk

import csv
import psutil


def getBCdata(start='2017-01-01', end=None):
    return CoinDesk.get_historical_data_as_dict(start, end)

# example usage
ts = getBCdata()

with open('Bitcoin2017.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in ts.items():
       writer.writerow([key, value])

