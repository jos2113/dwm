import numpy as np # linear algebra
import pandas as pd # data processing
import apyori
from apyori import apriori

data = pd.read_csv("Groceries_dataset.csv")
data.head()

data.isnull().any()

print("Total number of unique products are:", len(data['itemDescription'].unique()))

products = data['itemDescription'].unique()

dummy = pd.get_dummies(data['itemDescription'])

data.drop(['itemDescription'], inplace =True, axis=1)

data = data.join(dummy)

data.head()

data1 = data.groupby(['Member_number', 'Date'])[products[:]].sum()
data1 = data1.reset_index()[products]

data1.head()

def product_names(x):
    for product in products:
        if x[product] >0:
            x[product] = product
    return x

data1 = data1.apply(product_names, axis=1)
data1.head()

print("Total Number of Transactions:", len(data1))

x = data1.values
x = [sub[~(sub==0)].tolist() for sub in x if sub [sub != 0].tolist()]
transactions = x
transactions[0:10]

rules = apriori(transactions, min_support = 0.00050, min_confidence = 0.05, min_lift =
3, max_length = 2, target = "rules")
association_results = list(rules)
print(association_results[0])

for item in association_results:
    pair = item[0]
    items = [x for x in pair]

print("Rule : ", items[0], " -> " + items[1])
print("Support : ", str(item[1]))
print("Confidence : ",str(item[2][0][2]))

print("Lift : ", str(item[2][0][3]))
