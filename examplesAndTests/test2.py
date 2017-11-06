import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
hour, direction = np.meshgrid(np.arange(24), np.arange(1, 3))
df = pd.DataFrame({"hour": hour.flatten(), "direction": direction.flatten()})
df["hourly_avg_count"] = np.random.randint(14, 30, size=len(df))

plt.figure(figsize=(12, 8))
sns.tsplot(df, time='hour', unit="direction",
           condition='direction', value='hourly_avg_count')
plt.show()
