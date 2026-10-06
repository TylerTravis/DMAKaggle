import kagglehub
import pandas as pd
import numpy as np
import random
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Download latest version
# path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

test_passenger_ids = test_df['PassengerId']

def process_data(df):
	# Quantification and One-Hots

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

	# Feature Engineering
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	# df['IsAlone'] = df['FamilySize'].apply(lambda x: int(x == 1))
	# df['FarePerPerson'] = df['Fare'] / df['FamilySize']
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 15))

	# Imputation
	df['Age'] = df['Age'].fillna(df['Age'].median())
	df['Fare'] = df['Fare'].fillna(df['Fare'].median())

	return df

X_train = process_data(train_df)
X_test = process_data(test_df)

cols_to_drop = ['Name', 'Title', 'Cabin', 'Embarked', 'Ticket']

# Test & Train
X_train = X_train.drop(columns=['Survived', 'PassengerId'] + cols_to_drop)
Y_train = train_df['Survived']

X_test = X_test.drop(columns=['PassengerId'] + cols_to_drop)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Validation Split
X_train, X_val, Y_train, Y_val = train_test_split(
	X_train, Y_train, test_size=0.15, random_state=RANDOM_SEED
)

# Classifier
params = {
	'max_depth': 4,
	'learning_rate': 0.05,
	'n_estimators': 130,
	'min_child_weight': 5,
	'subsample': 0.75,
	'colsample_bytree': 0.75,
	'gamma': 0.175,
	'reg_alpha': 0.055,
	'reg_lambda': 1.5,
	'random_state': RANDOM_SEED,
	'eval_metric': 'logloss'
}

param_grid = {
	'max_depth': [3, 4],
	'learning_rate': [0.025, 0.05, 0.075],
	'n_estimators': [100, 125, 150],
	'min_child_weight': [4, 5, 6],
	'subsample': [0.6, 0.7, 0.8, 0.9],
	'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
	'gamma': [0, 0.1, 0.25, 0.5],
	'reg_alpha': [0, 0.01, 0.1],
	'reg_lambda': [1, 1.5, 2]
}

bst = xgb.XGBClassifier(**params)

n_iter = 125

rs_bst = RandomizedSearchCV(
	estimator=bst,
	param_distributions=param_grid,
	n_iter=n_iter,
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

rs_bst = rs_bst.best_estimator_

# Train Preds and Acc
train_pred = rs_bst.predict(X_train)
train_acc = rs_bst.score(X_train, Y_train)
print("Train Accuracy: ", train_acc)

# Validation Preds and Acc
val_acc = rs_bst.score(X_val, Y_val)
print("Validation Accuracy: ", val_acc)

# Recombine Training and Validation Data
X_train = pd.concat([X_train, X_val])
Y_train = pd.concat([Y_train, Y_val])

rs_bst.fit(X_train, Y_train)

# Recombined Acc
comb_acc = rs_bst.score(X_train, Y_train)
print("Combined Accuracy: ", comb_acc)

# Prediction Probability Threshold
val_probs = rs_bst.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_acc = 0

# Test thresholds 0.15 to 0.85
for thresh in np.arange(0.15, 0.85, 0.01):
	preds = (val_probs >= thresh).astype(int)
	acc = np.mean(preds == Y_val)
	if acc > best_acc:
		best_acc = acc
		best_thresh = thresh

test_probs = rs_bst.predict_proba(X_test)[:, 1]
submission_preds = (test_probs >= best_thresh).astype(int)

print('Validation Accuracy with Threshold Adjustment: ', best_acc)

# Export Submission Data
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
