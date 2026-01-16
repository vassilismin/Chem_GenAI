import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lipophilicity_astrazeneca.tab", sep="\t")
# Get statistics on the target variable
df["Y"].describe()

# Plot the distribution of the target variable
plt.hist(df["Y"], bins=30, edgecolor="k")
plt.title("Distribution of Lipophilicity (Y)")
plt.xlabel("Lipophilicity (Y)")
plt.ylabel("Frequency")
plt.savefig("data/lipophilicity_distribution.png", dpi=150)
print("Plot saved to data/lipophilicity_distribution.png")

# Check the number of molecules with logP between 1.5 and 3.0
count_in_range = df[(df["Y"] >= 1.5) & (df["Y"] <= 3.0)].shape[0]
print(f"Number of molecules with logP between 1.5 and 3.0: {count_in_range}")
