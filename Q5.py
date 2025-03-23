import pandas as pd

# Load dataset
file_path = "/mnt/data/Groceries_dataset.csv"
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())


print(df.info())  # Check column names & data types
print(df.nunique())  # Unique values in each column


# Group items by transaction (Member_number)
transactions = df.groupby(['Member_number'])['itemDescription'].apply(list).tolist()

print(transactions[:5])  # View first few transactions


from mlxtend.preprocessing import TransactionEncoder

# Encode transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Display transformed dataset
print(df_encoded.head())


from mlxtend.frequent_patterns import apriori

# Find frequent itemsets with minimum support of 0.01
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Display frequent itemsets
print(frequent_itemsets.head())


from mlxtend.frequent_patterns import association_rules

# Generate rules with minimum confidence of 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display important columns
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


# Strong Rules
strong_rules = rules[(rules['confidence'] > 0.6) & (rules['lift'] > 1.2)]

# Display strong rules
print(strong_rules)


import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of Support vs Confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(x=rules['support'], y=rules['confidence'], size=rules['lift'], hue=rules['lift'], palette="coolwarm", edgecolors="k", alpha=0.8)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence (Association Rules)")
plt.show()
