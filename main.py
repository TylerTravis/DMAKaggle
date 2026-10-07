# General
import catboost
import kagglehub
import pandas as pd
import numpy as np
import random

# SKLearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Download latest version
# path = kagglehub.competition_download('dma-26-kaggle-competition')

train_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/train.csv')
test_df = pd.read_csv('/Users/johnd/.cache/kagglehub/competitions/dma-26-kaggle-competition/test.csv')

test_passenger_ids = test_df['PassengerId']

#Imputation Values
train_age_median = train_df['Age'].median()
train_fare_median = train_df['Fare'].median()
train_embarked_mode = train_df['Embarked'].mode()[0]

def process_data(df):
	# Quantification and One-Hots

	# Imputation
	df['Age'] = df['Age'].fillna(train_age_median)
	df['Fare'] = df['Fare'].fillna(train_fare_median)
	df['Embarked'] = df['Embarked'].fillna(train_embarked_mode)

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

master_params = {
	'random_generations': 125,
	'max_depth': [3, 4],
	'learning_rate': [0.025, 0.05, 0.075],
	'n_estimators': [100, 110, 120],
	'min_child_weight': [4, 5, 6],
	'subsample': [0.6, 0.7, 0.8, 0.9],
	'l1_leaf_reg': [0, 0.01, 0.1],
	'l2_leaf_reg': [1, 1.5, 2, 2.5, 3.0]
}

# XGB Classifier
xgb_params = {
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

xgb_param_grid = {

	'max_depth': master_params['max_depth'],
	'learning_rate': master_params['learning_rate'],
	'n_estimators': master_params['n_estimators'],
	'min_child_weight': master_params['min_child_weight'],
	'subsample': master_params['subsample'],
	'colsample_bytree': master_params['subsample'],
	'gamma': [0, 0.1, 0.25, 0.5],
	'reg_alpha': master_params['l1_leaf_reg'],
	'reg_lambda': master_params['l2_leaf_reg']
}

xgb_clf = xgb.XGBClassifier(**xgb_params)

rs_xgb_clf = RandomizedSearchCV(
	estimator=xgb_clf,
	param_distributions=xgb_param_grid,
	n_iter=master_params['random_generations'],
	cv=5,
	scoring='accuracy',
	n_jobs=-1,
	random_state=RANDOM_SEED,
	# return_train_score=True
)

rs_xgb_clf.fit(X_train, Y_train)

# CatBoost Classifier
cat_params = {
	'objective': 'Logloss',
	'random_state': RANDOM_SEED,
	'eval_metric': 'Accuracy',
	'iterations': 125,
	'min_data_in_leaf': 4,
	'learning_rate': 0.05,
	'l2_leaf_reg': 3.0,
	'subsample': 0.66,
	'depth': 3,
	'verbose': 0
}

cat_param_grid = {
	'objective': ['Logloss'],
	'eval_metric': ['Accuracy'],
	'depth': master_params['max_depth'],
	'learning_rate': master_params['learning_rate'],
	'iterations': master_params['n_estimators'],
	'min_data_in_leaf': master_params['min_child_weight'],
	'subsample':master_params['subsample'],
	'l2_leaf_reg': master_params['l2_leaf_reg'],
	'random_state': [RANDOM_SEED],
}

cat_clf = cb.CatBoostClassifier(**cat_params)

rs_cat_clf = RandomizedSearchCV(
	estimator=cat_clf,
	param_distributions=cat_param_grid,
	n_iter=master_params['random_generations'],
	cv=5,
	scoring='accuracy',
	n_jobs=-1,
	random_state=RANDOM_SEED,
	# return_train_score=True,
	verbose=0
)

rs_cat_clf.fit(X_train, Y_train)

# LightGBM Classifier
light_params = {
	'max_depth': 3,
	'min_gain_to_split': 0.25,
	'n_estimators': 100,
	'max_bin': 30,
	'learning_rate': 0.09,
	'random_state': [RANDOM_SEED],
	'verbose': -1
}

light_param_grid = {
	'max_depth': master_params['max_depth'],
	'min_gain_to_split': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
	'n_estimators': master_params['n_estimators'],
	'max_bin': [10, 20, 30, 40, 50],
	'learning_rate':master_params['learning_rate'],
	'random_state': [RANDOM_SEED]
}

light_clf = lgb.LGBMClassifier(**light_params)

rs_light_clf = RandomizedSearchCV(
	estimator=light_clf,
	param_distributions=light_param_grid,
	n_iter=master_params['random_generations'],
	cv=5,
	scoring='accuracy',
	n_jobs=-1,
	random_state=RANDOM_SEED,
	# return_train_score=True,
	verbose=0
)

rs_light_clf.fit(X_train, Y_train)

# Threshold Optimization
xgb_opt_thresh = 0
cat_opt_thresh = 0
light_opt_thresh = 0
ensemble_thresh = 0

# Model Accuracies
xgb_val_acc = 0
cat_val_acc = 0
light_val_acc = 0
ens_val_acc = 0

# Ensemble
eclf = VotingClassifier(
	estimators=[
		('xgb', rs_xgb_clf.best_estimator_),
		('cat', rs_cat_clf.best_estimator_),
		('light', rs_light_clf.best_estimator_)
	],
	voting='soft',
	weights=[1, 1, 1]
)

for clf in [rs_xgb_clf, rs_cat_clf, rs_light_clf, eclf]:
	# Export hyperparameter data
	# perf_df = pd.DataFrame(rs_xgb_clf.cv_results_)
	# perf_df.to_csv('optimization.csv', index=False)

	if clf == rs_xgb_clf:
		print("\nXGB Classifier\n")
	elif clf == rs_cat_clf:
		print("\nCatBoost Classifier\n")
	elif clf == rs_light_clf:
		print("\nLightGBM Classifier\n")
	else:
		print("\nEnsemble Classifier\n")

	if clf == eclf:
		clf.fit(X_train, Y_train)


	# Train Preds and Acc
	train_pred = clf.predict(X_train)
	train_acc = clf.score(X_train, Y_train)
	print("Train Accuracy: ", train_acc)

	# Validation Preds and Acc
	val_acc = clf.score(X_val, Y_val)
	print("Validation Accuracy: ", val_acc)

	# Pred Prob Threshold
	val_probs = clf.predict_proba(X_val)[:, 1]

	best_thresh = 0.5
	best_acc = 0

	# Test thresholds 0.1 to 0.9
	for thresh in np.arange(0.3, 0.7, 0.005):
		preds = (val_probs >= thresh).astype(int)
		acc = np.mean(preds == Y_val)
		if acc > best_acc:
			best_acc = acc
			best_thresh = thresh
	print('Validation Accuracy with Threshold Adjustment: ', best_acc)

	# Assign Optimal Thresholds
	if clf == rs_xgb_clf:
		xgb_opt_thresh = best_thresh
		xgb_val_acc = best_acc
	elif clf == rs_cat_clf:
		cat_opt_thresh = best_thresh
		cat_val_acc = best_acc
	elif clf == rs_light_clf:
		light_opt_thresh = best_thresh
		light_val_acc = best_acc
	elif clf == eclf:
		ens_opt_thresh = best_thresh
		ens_val_acc = best_acc

# Recombine Training and Validation Data
X_train = pd.concat([X_train, X_val])
Y_train = pd.concat([Y_train, Y_val])

best_val_acc = max([xgb_val_acc, cat_val_acc, light_val_acc, ens_val_acc])
opt_thresh = 0

# Select Best Model
if best_val_acc == xgb_val_acc:
	clf = rs_xgb_clf.best_estimator_
	opt_thresh = xgb_opt_thresh
elif best_val_acc == cat_val_acc:
	clf = rs_cat_clf.best_estimator_
	opt_thresh = cat_opt_thresh
elif best_val_acc == light_val_acc:
	clf = rs_light_clf.best_estimator_
	opt_thresh = light_opt_thresh
elif best_val_acc == ens_val_acc:
	clf = eclf
	opt_thresh = ens_opt_thresh

clf.fit(X_train, Y_train)

# Final Subission Preds
test_probs = clf.predict_proba(X_test)[:, 1]
submission_preds = (test_probs >= opt_thresh).astype(int)

# Export Submission Data
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
