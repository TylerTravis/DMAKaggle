import kagglehub
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

all_data = pd.concat([train_df, test_df])

# Imputation Values
train_age_median = train_df['Age'].median()
train_fare_median = train_df['Fare'].median()
train_embarked_mode = train_df['Embarked'].mode()[0]

# Group Survival Rate
def calculate_group_survival(df):
	copy = df.copy()
	copy['LastName'] = copy['Name'].str.extract(r'([A-Za-z]+),', expand=False)

	def get_companion_survival_rate(passenger_id, last_name, ticket):
		companions = copy[
			((copy['LastName'] == last_name) | (copy['Ticket'] == ticket)) &
			(copy.index != passenger_id)
			]

		companions = companions.dropna(subset=['Survived'])

		if len(companions) > 0:
			return companions['Survived'].mean()
		else:
			return -1

	return copy.apply(lambda x: get_companion_survival_rate(x.name, x['LastName'], x['Ticket']), axis=1)

all_data['GroupSurvivalRate'] = calculate_group_survival(all_data)
train_df['GroupSurvivalRate'] = all_data['GroupSurvivalRate'].iloc[:len(train_df)]
test_df['GroupSurvivalRate'] = all_data['GroupSurvivalRate'].iloc[len(train_df):]

def process_data(df):
	# Imputation
	df['Age'] = df['Age'].fillna(train_age_median)
	df['Fare'] = df['Fare'].fillna(train_fare_median)
	df['Embarked'] = df['Embarked'].fillna(train_embarked_mode)

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
		['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Mrs'], 'Rare')

	# Embarked
	df['Embarked'] = df['Embarked'].astype('category').cat.codes

	# Cabin
	df['Cabin'] = df['Cabin'].str.extract(r'([A-Za-z])', expand=False).astype('category').cat.codes

	# Family Survival Detection

	# Added Features
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 12))
	df['HasCabin'] = (df['Cabin'] != -1).astype(int)
	df['RareTitle'] = df['Title'].apply(lambda x: int(x == 'Rare'))
	df['ExpensiveFare'] = df['Fare'].apply(lambda x: int(x > 40))
	df['WhyTFDoesThisWork'] = df['FamilySize'].apply(lambda x: int(x >= 1))
	# df['IsChildAndFamilyMemberSurvived'] = df.apply(lambda row: int(row['IsChild'] and row['GroupSurvivalRate'] > 0), axis=1)

	return df

X_train = process_data_for_decision_tree(train_df)
X_test = process_data_for_decision_tree(test_df)

# print(X_train.head())

cols_for_corr = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'FamilySize', 'IsChild', 'HasCabin', 'RareTitle', 'ExpensiveFare', 'GroupSurvivalRate', 'WhyTFDoesThisWork']

for col in X_train.columns:
	print('\nVALUE COUNTS\n')
	print(X_train[col].value_counts())

	if col in cols_for_corr:
		print('\nCORRELATION WITH SURVIVAL')
		print(X_train[col].corr(X_train['Survived']))