import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./Fantasy Football.csv')

# df = df.drop(columns=['PPR','PPG','PPRPG','PosRank'], axis=1)

def add_next_year(row):
    name = row['Name']
    year = row['Year']
    next_year = int(year) + 1
    result = df.loc[(df['Name'] == name) & (df['Year'] == next_year)]['FantPt']
    value = result.values[0] if not result.empty else 0
    return value
        

df['NextYear'] = df.apply(add_next_year, axis=1)
df = df[(df[['NextYear']] != 0).any(axis=1)]
df = df.dropna(subset=['NextYear'])
# print(df['NextYear'].value_counts().reset_index(name='Counts'))

# One hot encoding
df = pd.get_dummies(df, columns=['Name', 'Year', 'Tm', 'FantPos', 'Age'], drop_first=True, dtype='int')
# df = df.drop(columns=['Name', 'Tm', 'FantPos'])
# df = df.join(one_hot)

# plt.hist(df['NextYear'])
# plt.show()
df = df.fillna(0)

col_to_join = df.pop('NextYear')

# col2_to_join = df.pop('Name')

# normalize columns
# df = df.apply(lambda x: (x - x.min())/(x.max() - x.min()), axis=0)

df = pd.concat([df, col_to_join], axis=1)
# df = pd.concat([df, col2_to_join], axis=1)

df.to_csv('draft_prediction_denormalized.csv', index=False, encoding='utf-8')