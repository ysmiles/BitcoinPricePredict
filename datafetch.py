from exchanges.coindesk import CoinDesk


def getBCdata(start='2017-01-01', end=None):
    return CoinDesk.get_historical_data_as_dict(start, end)

# print(CoinDesk.get_historical_data_as_dict(start='2017-10-10', end=None))
