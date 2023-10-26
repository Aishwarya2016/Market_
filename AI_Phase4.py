import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel("Assignment-1_Data.xlsx")
df.info()
df.isnull().sum
df.dropna(subset=["Itemname"],inplace=True)
Quantity <=0
df = df[df["Quantity"]>0]
df.isnull().sum()
df['CustomerID'].fillna(99999, inplace=True)
df_encoded = pd.read_csv('transaction_data_encoded.csv')
frequent_itemsets = apriori(df_encoded, min_support=0.007, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("ASSOCIATION RULES:")
print(rules.head)
plt.figure(figsize=(12, 8))
sns.scatterplot(x="support", y="confidence", size="lift", data=rules, hue="lift", palette="viridis", sizes=(20, 200))
plt.title('Market Basket Analysis - Support vs. Confidence (Size = Lift)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.legend(title='Lift', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()