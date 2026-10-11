import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Download latest version
# path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

test_passenger_ids = test_df['PassengerId']

# Imputation Values
train_age_median = train_df['Age'].median()
train_fare_median = train_df['Fare'].median()
train_embarked_mode = train_df['Embarked'].mode()[0]

# Multiple Member Survival Rates
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

def calculate_same_ticket_survival(df):
	copy = df.copy()
	copy['LastName'] = copy['Name'].str.extract(r'([A-Za-z]+),', expand=False)

	def get_companion_survival_rate(passenger_id, last_name, ticket):
		companions = copy[
			((copy['Ticket'] == ticket)) &
			(copy.index != passenger_id)
			]

		companions = companions.dropna(subset=['Survived'])

		if len(companions) > 0:
			return companions['Survived'].mean()
		else:
			return -1

	return copy.apply(lambda x: get_companion_survival_rate(x.name, x['LastName'], x['Ticket']), axis=1)

all_data = pd.concat([train_df, test_df])

all_data['GroupSurvivalRate'] = calculate_group_survival(all_data)
all_data['SameTicketSurvivalRate'] = calculate_same_ticket_survival(all_data)

train_df['GroupSurvivalRate'] = all_data['GroupSurvivalRate'].iloc[:len(train_df)]
test_df['GroupSurvivalRate'] = all_data['GroupSurvivalRate'].iloc[len(train_df):]

train_df['SameTicketSurvivalRate'] = all_data['SameTicketSurvivalRate'].iloc[:len(train_df)]
test_df['SameTicketSurvivalRate'] = all_data['SameTicketSurvivalRate'].iloc[len(train_df):]

def process_data(df):
	df = df.copy()
	# Imputation
	df['Age'] = df['Age'].fillna(train_age_median)
	df['Fare'] = df['Fare'].fillna(train_fare_median)

	# Quantification and One-Hots

	# Sex
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

	# Class
	pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass', dtype=int)
	df = pd.concat([df, pclass_dummies], axis=1)

	# Name / Title
	df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
	df['Title'] = df['Title'].replace(['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Miss', 'Mlle', 'Ms', 'Mme', 'Mrs'], 'Rare')

	title_dummies = pd.get_dummies(df['Title'], prefix='Title', dtype=int)
	df = pd.concat([df, title_dummies], axis=1)

	# Embarked
	embarked_dummies = pd.get_dummies(df['Embarked'].str[0], prefix='Embarked', dummy_na=True, dtype=int)
	df = pd.concat([df, embarked_dummies], axis=1)

	# Cabin
	cabin_dummies = pd.get_dummies(df['Cabin'].str[0], prefix='Cabin', dummy_na=True, dtype=int)
	df = pd.concat([df, cabin_dummies], axis=1)

	# Logs
	df['Log_Fare'] = np.log(df['Fare'] + 0.5)
	df['Log_Age'] = np.log(df['Age'] + 0.5)

	# Feature Engineering
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 7))
	df['IsYouth'] = df['Age'].apply(lambda x: int(7 <= x < 14))
	df['HasCabin'] = df['Cabin'].notna().astype(int)
	df['RareTitle'] = df['Title'].apply(lambda x: int(x == 'Rare'))
	df['ExpensiveFare'] = df['Fare'].apply(lambda x: int(x > 39))
	df['IsChildAndFamilyMemberSurvived'] = df.apply(lambda row: int(row['IsChild'] and row['GroupSurvivalRate'] > 0), axis=1)

	return df

def process_data_for_decision_tree(df, model=''):
	df = df.copy()
	# Sex
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

	# Name / Title
	df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
	df['Title'] = df['Title'].replace(
		['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Mrs', 'Miss', 'Mlle', 'Mme' 'Ms'], 'Rare')

	# Embarked
	df['Embarked'] = df['Embarked'].astype('category').cat.codes

	# Cabin
	# df['Cabin'] = df['Cabin'].str.extract(r'([A-Za-z])', expand=False).astype('category').cat.codes

	# Added Features
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 6))
	df['HasCabin'] = df['Cabin'].notna().astype(int)
	df['RareTitle'] = df['Title'].apply(lambda x: int(x == 'Rare'))
	df['ExpensiveFare'] = df['Fare'].apply(lambda x: int(x > 39))
	df['IsChildAndGroupMemberSurvived'] = df.apply(lambda row: int(row['IsChild'] and row['GroupSurvivalRate'] > 0), axis=1)

	return df

# DECISION TREE Data
X_train_DT = process_data_for_decision_tree(train_df)
X_test_DT = process_data_for_decision_tree(test_df)

cols_to_drop_DT = ['Name', 'Title', 'Cabin', 'Ticket', 'Fare']

# Test & Train
X_train_DT = X_train_DT.drop(columns=['Survived', 'PassengerId'] + cols_to_drop_DT)
Y_train = train_df['Survived']

X_test_DT = X_test_DT.drop(columns=['PassengerId'] + cols_to_drop_DT)

# X_train_DT = X_train_DT.add_suffix('_DT')
# X_test_DT = X_test_DT.add_suffix('_DT')

# GEOMETRIC DATA
X_train_G = process_data(train_df)
X_test_G = process_data(test_df)

cols_to_drop_G = ['Name', 'Title', 'Cabin', 'Embarked', 'Pclass', 'Ticket', 'Fare', 'Age']

# Test & Train
X_train_G = X_train_G.drop(columns=['Survived', 'PassengerId'] + cols_to_drop_G)

X_test_G = X_test_G.drop(columns=['PassengerId'] + cols_to_drop_G)

# X_train_G = X_train_G.add_suffix('_G')
# X_test_G = X_test_G.add_suffix('_G')

# Feature Lists
# features_DT = X_train_DT.columns.tolist()
# features_G = X_train_G.columns.tolist()
#
# # Merge DataFrames
# X_train_ALL = pd.concat([X_train_DT, X_train_G], axis=1)
# X_test_ALL = pd.concat([X_test_DT, X_test_G], axis=1)
#
# # Validation Split
X_train_CV, X_val, Y_train_CV, Y_val = train_test_split(
    X_train_DT, Y_train, test_size=0.15, random_state=RANDOM_SEED
)
