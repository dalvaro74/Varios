import pandas as pd
df = pd.DataFrame([[0, 1, 2, 3, 1],
 [3, 1, 0, 1, 1],
 [0, 1, 0, 3, 0],
 [0, 2, 0, 1, 2],
 [0, 2, 0, 3, 2]], columns=['color1', 'color2', 'color3', 'color4', 'color5'])

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Get all the unique values if we don't have them
unique_values = pd.unique(df.values.ravel()) 

type([unique_values]*len(df))

list_cat =[[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3]]
ohe = OneHotEncoder(categories=list_cat,sparse=False)
#ohe = OneHotEncoder(sparse=False)
#print(ohe.fit(df[['color1', 'color2', 'color3', 'color4', 'color5']]).transform(df[['color1', 'color2', 'color3', 'color4', 'color5']]))
#print(ohe.fit_transform( df[['color1', 'color2', 'color3', 'color4', 'color5']]))
print(ohe.fit_transform( df))
#encoded = pd.DataFrame(ohe.fit_transform( df[['color1']]), columns=ohe.get_feature_names(df.columns))
#print(df)
#print(encoded)