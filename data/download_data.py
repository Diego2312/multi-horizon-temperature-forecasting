import pandas as pd

url = "https://www.ncei.noaa.gov/data/daily-summaries/access/IT000016239.csv"
df = pd.read_csv(url)
df.to_csv("data/Raw_dataset.csv", index=False)
print("Downloaded and saved to data/Raw_dataset.csv")
