import kagglehub
import pandas as pd
import numpy as np
import random
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
OPTIMIZING = True

random.seed(RANDOM_SEED)

# Download latest version
# path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

test_passenger_ids = test_df['PassengerId']

def process_data(df):
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

# Validation Split
X_train, X_val, Y_train, Y_val = train_test_split(
	X_train, Y_train, test_size=0.15, random_state=RANDOM_SEED
)

# Classifier
train_matrix = xgb.DMatrix(X_train, label=Y_train)
test_matrix = xgb.DMatrix(X_test)

params = {
	'max_depth': 4,
	'learning_rate': 0.05,
	'n_estimators': 150,
	'min_child_weight': 3,
	'subsample': 0.8,
	'colsample_bytree': 0.8,
	'gamma': 0.1,
	'random_state': RANDOM_SEED,
	'eval_metric': 'logloss'
}

param_grid = {
	'max_depth': [3, 4, 5],
	'learning_rate': [0.01, 0.02, 0.05, 0.1],
	'n_estimators': [100, 150, 200, 300],
	'min_child_weight': [3, 5, 7],
	'subsample': [0.6, 0.7, 0.8, 0.9],
	'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
	'gamma': [0, 0.1, 0.25, 0.5],
	'reg_alpha': [0, 0.01, 0.1],
	'reg_lambda': [1, 1.5, 2]
}

bst = xgb.XGBClassifier(**params)



if OPTIMIZING:
	top_perf_refine_prop = 0.1
	n_iter = 1000

	rs_bst = RandomizedSearchCV(
		estimator=bst,
		param_distributions=param_grid,
		n_iter=125,
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		random_state=RANDOM_SEED,
		return_train_score=True
	)

	rs_bst.fit(X_train, Y_train)

	# Export hyperparameter data
	perf_df = pd.DataFrame(rs_bst.cv_results_)
	perf_df.to_csv('optimization.csv', index=False)


bst.fit(X_train, Y_train)

# Train Preds and Acc
train_pred = bst.predict(X_train)
train_acc = bst.score(X_train, Y_train)
print("Train Accuracy: ", train_acc)

if OPTIMIZING:
	# CV Acc
	train_cv_acc = rs_bst.best_score_
	print("Cross-Validation Accuracy: ", train_cv_acc)

# Validation Preds and Acc
best_model = rs_bst.best_estimator_
val_acc = best_model.score(X_val, Y_val)
print("Validation Accuracy: ", val_acc)


# Export Submission Data
submission_preds = bst.predict(X_test)
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
