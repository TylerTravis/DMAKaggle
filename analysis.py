import kagglehub
import pandas as pd
import numpy as np

path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def process_data(df):
	# Imputation
	# df['Age'] = df['Age'].fillna(train_age_median)
	# df['Fare'] = df['Fare'].fillna(train_fare_median)
	# df['Embarked'] = df['Embarked'].fillna(train_embarked_mode)

	# Quantification and One-Hots

	# Sex
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

	# Class
	pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass', dtype=int)
	df = pd.concat([df, pclass_dummies], axis=1)

	# Name / Title
	df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
	df['Title'] = df['Title'].replace(['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'], 'Rare')
	df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
	df['Title'] = df['Title'].replace('Mme', 'Mrs')

	title_dummies = pd.get_dummies(df['Title'], prefix='Title', dtype=int)
	df = pd.concat([df, title_dummies], axis=1)

	# Embarked
	embarked_dummies = pd.get_dummies(df['Embarked'].str[0], prefix='Embarked', dummy_na=True, dtype=int)
	df = pd.concat([df, embarked_dummies], axis=1)

	# Cabin
	cabin_dummies = pd.get_dummies(df['Cabin'].str[0], prefix='Cabin', dummy_na=True, dtype=int)
	df = pd.concat([df, cabin_dummies], axis=1)

	# Fare
	df['Fare'] = np.log1p(df['Fare'])

	# Feature Engineering
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	# df['IsAlone'] = df['FamilySize'].apply(lambda x: int(x == 1))
	# df['FarePerPerson'] = df['Fare'] / df['FamilySize']
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 15))

	return df

def process_data_for_decision_tree(df):
	# Sex
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

	# Name / Title
	df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
	df['Title'] = df['Title'].replace(
		['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'], 'Rare')

	df['LastName'] = df['Name'].str.extract(r'([A-Za-z]+),', expand=False)

	# Added Features
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 15))

	return df

X_train = process_data_for_decision_tree(train_df)
X_test = process_data_for_decision_tree(test_df)

# print(X_train.head())
# print(X_test.head())



for col in train_df.columns:
	print(train_df[col].value_counts())

	# Survival Correlations
