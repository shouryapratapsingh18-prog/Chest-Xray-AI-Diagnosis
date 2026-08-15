import pandas as pd

# Path to your actual sample labels CSV file
df = pd.read_csv(r"C:\Users\shour\Desktop\DATA\archive\sample\sample_labels.csv")

# Jis image ka label check karna hai uska filename daalein
img_name = "00000099_003.png"
result = df[df["Image Index"] == img_name]
print(result[["Image Index", "Finding Labels"]])