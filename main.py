import kagglehub
import pandas as pd
import numpy as np
import random

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Download latest version
# path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

# Models: MLP - RF - HGB
RANDOM_SEED = 42
MODEL = 'HGB'
OPTIMIZING = True

random.seed(RANDOM_SEED)
test_passenger_ids = test_df['PassengerId']

def process_data(df):
	if MODEL == 'MLP':
		# Impute Missing Values
		df['Age'] = df['Age'].fillna(df['Age'].mean())
		df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

	# Quantitative and One-Hot Refactor

	# Class
	class_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass', dtype=int)
	df = pd.concat([df, class_dummies], axis=1)
	df = df.rename(columns={'Pclass_1.0': 'Pclass_1', 'Pclass_2.0': 'Pclass_2', 'Pclass_3.0': 'Pclass_3'})

	# Sex
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

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

	# Logs
	df['Log_Fare'] = np.log(df['Fare'] + 0.5)
	df['Log_Age'] = np.log(df['Age'] + 0.5)

	# Returning the dataframe allows us to safely overwrite the old ones!
	return df

X_train = process_data(train_df)
X_test = process_data(test_df)

cols_to_drop = ['Name', 'Title', 'Cabin', 'Embarked', 'Pclass', 'Ticket', 'Fare', 'Age']

# Test & Train
X_train = X_train.drop(columns=['Survived', 'PassengerId'] + cols_to_drop)
Y_train = train_df['Survived']

X_test = X_test.drop(columns=['PassengerId'] + cols_to_drop)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale Data
scaler = StandardScaler()
cols_to_scale = ['SibSp', 'Parch', 'Log_Fare', 'Log_Age']

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


if MODEL == 'MLP':
	clf = MLPClassifier(hidden_layer_sizes=(22, 22), solver='lbfgs', alpha=0.0075, activation='relu', random_state=RANDOM_SEED, max_iter=1000)

	# Layers and Node Counts
	node_sizes = range(19, 26)
	hidden_layer_sizes = []

	# Randomly Generate Various Layer Sizes
	for _ in range(100):
		num_layers = random.randint(2, 3)
		architecture = tuple(random.choice(node_sizes) for _ in range(num_layers))
		hidden_layer_sizes.append(architecture)

	param_grid = {
		'hidden_layer_sizes': hidden_layer_sizes,
		'activation': ['relu'],
		'alpha': [0.05, 0.075, 0.01],
		'solver': ['lbfgs']
	}
elif MODEL == 'RF':
	clf = RandomForestClassifier(random_state=RANDOM_SEED)

	n_estimators = [i * 50 for i in range(1, 11)]
	max_depth = [i for i in range(1, 21)]
	min_samples_split = [i for i in range(2, 11)]
	min_samples_leaf = [i for i in range(1, 6)]

	param_grid = {
		'n_estimators': n_estimators,
		'max_depth': max_depth,
		'min_samples_split': min_samples_split,
		'min_samples_leaf': min_samples_leaf
	}
elif MODEL == 'HGB':
	clf = HistGradientBoostingClassifier(learning_rate=0.02, max_iter=300, max_depth= 13, l2_regularization=0.2, random_state=RANDOM_SEED)

	learning_rates = [i * 0.005 for i in range(4, 11)]
	max_iterations = [i * 50 for i in range(4, 9)]
	max_depth = [i for i in range(9, 16)]
	l2_reg = [i * 0.05 for i in range(1, 11)]

	param_grid = {
		'learning_rate': learning_rates,
		'max_iter': max_iterations,
		'max_depth': max_depth,
		'l2_regularization': l2_reg
	}

if OPTIMIZING:
	clf = RandomizedSearchCV(
		estimator=clf,
		param_distributions=param_grid,
		n_iter=1000,
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		random_state=RANDOM_SEED
	)

clf.fit(X_train, Y_train)

# Train Preds and Acc
train_pred = clf.predict(X_train)
train_acc = clf.score(X_train, Y_train)
print("Train Accuracy: ", train_acc)

if OPTIMIZING:
	# CV Acc
	train_cv_acc = clf.best_score_
	print("Cross-Validation Accuracy: ", train_cv_acc)

	# Export hyperparameter data
	results_df = pd.DataFrame(clf.cv_results_)
	results_df.to_csv('optimization.csv', index=False)

# Export Submission Data
submission_preds = clf.predict(X_test)
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
