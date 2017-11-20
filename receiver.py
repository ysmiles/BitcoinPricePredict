import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pymongo import MongoClient


# database client
client = MongoClient()
# will fetch data from this collection
collection = client.bitcoin_predict.parameters

# Fetch data
Y_test = np.array(collection.find({'name': 'Y_test'})[0]['val'])
y_pred = np.array(collection.find({'name': 'y_pred'})[0]['val'])
epochs = np.array(collection.find({'name': 'epochs'})[0]['val'])
mses = np.array(collection.find({'name': 'mses'})[0]['val'])
cpus = np.array(collection.find({'name': 'cpus'})[0]['val'])
mems = np.array(collection.find({'name': 'mems'})[0]['val'])

# Result
plt.figure()
plt.title("Predicted vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "o", markersize=10, label="Actual")
# plt.plot(pd.Series(np.ravel(Y_test)), "w.", markersize=10)
plt.plot(pd.Series(np.ravel(y_pred)), "*", markersize=10, label="Forecast")
plt.legend(loc="upper left")
plt.xlabel("Bitcoin prices of recent days")
plt.ylabel("Dollars")
# plt.axis([-1, 21, 4000, 6200])
plt.savefig("img/result.svg")

# Loss function
plt.figure()
plt.title("MSE vs Training Cycles", fontsize=14)
plt.plot(epochs, mses)
plt.legend(loc="upper left")
plt.xlabel("Cycles")
plt.ylabel("Mean square error")
plt.savefig("img/loss.svg")

# CPU usage
plt.figure()
plt.title("CPU usage vs Training Cycles", fontsize=14)
plt.plot(epochs, cpus)
plt.legend(loc="upper left")
plt.xlabel("Cycles")
plt.ylabel("CPU usage")
plt.savefig("img/cpu.svg")

# Memory usage
plt.figure()
plt.title("Memory usage vs Training Cycles", fontsize=14)
plt.plot(epochs, mems)
plt.legend(loc="upper left")
plt.xlabel("Cycles")
plt.ylabel("Memory usage")
plt.savefig("img/mem.svg")

plt.show()
