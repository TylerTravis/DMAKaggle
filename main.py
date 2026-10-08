# General
import catboost
import kagglehub
import pandas as pd
import numpy as np
import random

# SKLearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

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

all_data = pd.concat([train_df, test_df])

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
		['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'], 'Rare')

	# Embarked
	df['Embarked'] = df['Embarked'].astype('category').cat.codes

	# Cabin
	# df['Cabin'] = df['Cabin'].str.extract(r'([A-Za-z])', expand=False).astype('category').cat.codes

	# Family Survival Detection

	# Added Features
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 15))
	df['HasCabin'] = df['Cabin'].notnull().astype(int)
	df['RareTitle'] = df['Title'].apply(lambda x: int(x == 'Rare'))
	df['ExpensiveFare'] = df['Fare'].apply(lambda x: int(x > 40))
	df['IsNotAlone'] = df['FamilySize'].apply(lambda x: int(x == 1))

	return df

X_train = process_data_for_decision_tree(train_df)
X_test = process_data_for_decision_tree(test_df)

cols_to_drop = ['Name', 'Title', 'Cabin', 'Ticket', 'Fare', 'FamilySize']

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
	'max_depth': 3,
	'learning_rate': 0.05,
	'n_estimators': 70,
	'min_child_weight': 4,
	'subsample': 0.6,
	'colsample_bytree': 0.75,
	'l1_leaf_reg': 0.075,
	'l2_leaf_reg': 1,
	'random_state': RANDOM_SEED,
	'eval_metric': 'logloss'
}

master_param_grid = {
	'random_generations': 115,
	'max_depth': [3],
	'learning_rate': [0.075],
	'n_estimators': [7],
	'min_child_weight': [6, 7, 8],
	'subsample': [0.6, 0.7, 0.8],
	'l1_leaf_reg': [0, 0.05, 0.1],
	'l2_leaf_reg': [1.5, 2, 2.5]
}

# XGB Classifier
xgb_params = {
	'max_depth': master_params['max_depth'],
	'learning_rate': master_params['learning_rate'],
	'n_estimators': master_params['n_estimators'],
	'min_child_weight': master_params['min_child_weight'],
	'subsample': master_params['subsample'],
	'colsample_bytree': master_params['colsample_bytree'],
	'gamma': 0.175,
	'reg_alpha': master_params['l1_leaf_reg'],
	'reg_lambda': master_params['l2_leaf_reg'],
	'random_state': RANDOM_SEED,
	'eval_metric': master_params['eval_metric']
}

xgb_param_grid = {
	'max_depth': master_param_grid['max_depth'],
	'learning_rate': master_param_grid['learning_rate'],
	'n_estimators': master_param_grid['n_estimators'],
	'min_child_weight': master_param_grid['min_child_weight'],
	'subsample': master_param_grid['subsample'],
	'colsample_bytree': master_param_grid['subsample'],
	'gamma': [0, 0.1, 0.25, 0.5],
	'reg_alpha': master_param_grid['l1_leaf_reg'],
	'reg_lambda': master_param_grid['l2_leaf_reg']
}

xgb_clf = xgb.XGBClassifier(**xgb_params)

# CatBoost Classifier
cat_params = {
	'objective': 'Logloss',
	'random_state': RANDOM_SEED,
	'eval_metric': 'Accuracy',
	'iterations': master_params['n_estimators'],
	'min_data_in_leaf': master_params['min_child_weight'],
	'learning_rate': master_params['learning_rate'],
	'l2_leaf_reg': master_params['l2_leaf_reg'],
	'subsample': master_params['subsample'],
	'depth': master_params['max_depth'],
	'verbose': 0,
	'od_type': 'Iter'
}

cat_param_grid = {
	'objective': ['Logloss'],
	'eval_metric': ['Accuracy'],
	'depth': master_param_grid['max_depth'],
	'learning_rate': master_param_grid['learning_rate'],
	'iterations': master_param_grid['n_estimators'],
	'min_data_in_leaf': master_param_grid['min_child_weight'],
	'subsample':master_param_grid['subsample'],
	'l2_leaf_reg': master_param_grid['l2_leaf_reg'],
	'random_state': [RANDOM_SEED],
}

cat_clf = cb.CatBoostClassifier(**cat_params)

# LightGBM Classifier
light_params = {
	'task': 'predict',
	'boosting': 'gbdt',
	'data_sample_strategy': 'bagging',
	'objective': 'binary',
	'max_depth': master_params['max_depth'],
	'min_gain_to_split': 0.25,
	'n_estimators': master_params['n_estimators'],
	'max_bin': 30,
	'learning_rate': master_params['learning_rate'],
	'random_state': [RANDOM_SEED],
	'verbose': -1
}

light_param_grid = {
	'max_depth': master_param_grid['max_depth'],
	'min_gain_to_split': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
	'n_estimators': master_param_grid['n_estimators'],
	'max_bin': [10, 20, 30, 40, 50],
	'learning_rate':master_param_grid['learning_rate'],
	'random_state': [RANDOM_SEED]
}

light_clf = lgb.LGBMClassifier(**light_params)

# Ensemble
eclf = VotingClassifier(
	estimators=[
		('xgb', xgb_clf),
		('cat', cat_clf),
		('light', light_clf)
	],
	voting='soft',
	weights=[1, 1, 3]
)


OPTIMIZING = False
DEEP_OPTIMIZING = False

if OPTIMIZING:
	rs_xgb_clf = RandomizedSearchCV(
		estimator=xgb_clf,
		param_distributions=xgb_param_grid,
		n_iter=master_param_grid['random_generations'],
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		random_state=RANDOM_SEED,
	)

	rs_xgb_clf.fit(X_train, Y_train)

	rs_cat_clf = RandomizedSearchCV(
		estimator=cat_clf,
		param_distributions=cat_param_grid,
		n_iter=master_param_grid['random_generations'],
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		random_state=RANDOM_SEED,
		# return_train_score=True,
		verbose=0
	)

	rs_cat_clf.fit(X_train, Y_train)

	rs_light_clf = RandomizedSearchCV(
		estimator=light_clf,
		param_distributions=light_param_grid,
		n_iter=master_param_grid['random_generations'],
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		random_state=RANDOM_SEED,
		# return_train_score=True,
		verbose=0
	)

	rs_light_clf.fit(X_train, Y_train)

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

	# Model Accuracies
	xgb_val_acc = 0
	cat_val_acc = 0
	light_val_acc = 0
	ens_val_acc = 0

	for clf in [rs_xgb_clf, rs_cat_clf, rs_light_clf, eclf]:
		if clf == rs_xgb_clf:
			print("\nXGB Classifier")
		elif clf == rs_cat_clf:
			print("\nCatBoost Classifier")
		elif clf == rs_light_clf:
			print("\nLightGBM Classifier")
		else:
			print("\nEnsemble Classifier")

		if clf == eclf:
			clf.fit(X_train, Y_train)

		# Train Preds and Acc
		train_pred = clf.predict(X_train)
		train_acc = clf.score(X_train, Y_train)
		print("Train Accuracy: ", train_acc)

		# Validation Preds and Acc
		val_acc = clf.score(X_val, Y_val)
		print("Validation Accuracy: ", val_acc)

		if clf == rs_xgb_clf:
			xgb_val_acc = val_acc
		elif clf == rs_cat_clf:
			cat_val_acc = val_acc
		elif clf == rs_light_clf:
			light_val_acc = val_acc
		else:
			ens_val_acc = val_acc
elif DEEP_OPTIMIZING:
	# XGB Classifier
	gs_xgb_clf = GridSearchCV(
		estimator=xgb_clf,
		param_grid=xgb_param_grid,
		cv=5,
		scoring='accuracy',
		n_jobs=-1
	)
	gs_xgb_clf.fit(X_train, Y_train)

	# CatBoost Classifier
	gs_cat_clf = GridSearchCV(
		estimator=cat_clf,
		param_grid=cat_param_grid,
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		verbose=0
	)
	gs_cat_clf.fit(X_train, Y_train)

	# LightGBM Classifier
	gs_light_clf = GridSearchCV(
		estimator=light_clf,
		param_grid=light_param_grid,
		cv=5,
		scoring='accuracy',
		n_jobs=-1,
		verbose=0
	)
	gs_light_clf.fit(X_train, Y_train)

	# Ensemble using the best estimators from GridSearchCV
	eclf = VotingClassifier(
		estimators=[
			('xgb', gs_xgb_clf.best_estimator_),
			('cat', gs_cat_clf.best_estimator_),
			('light', gs_light_clf.best_estimator_)
		],
		voting='soft',
		weights=[1, 1, 1]
	)

	# Aggregated Results


	# Model Accuracies
	xgb_val_acc = 0
	cat_val_acc = 0
	light_val_acc = 0
	ens_val_acc = 0

	for clf in [gs_xgb_clf, gs_cat_clf, gs_light_clf, eclf]:
		if clf == gs_xgb_clf:
			print("\nXGB Classifier")
		elif clf == gs_cat_clf:
			print("\nCatBoost Classifier")
		elif clf == gs_light_clf:
			print("\nLightGBM Classifier")
		else:
			print("\nEnsemble Classifier")

		# Fit the ensemble (GridSearch models are already fit)
		if clf == eclf:
			clf.fit(X_train, Y_train)

		# Train Preds and Acc
		train_pred = clf.predict(X_train)
		train_acc = clf.score(X_train, Y_train)
		print("Train Accuracy: ", train_acc)

		# Validation Preds and Acc
		val_acc = clf.score(X_val, Y_val)
		print("Validation Accuracy: ", val_acc)

		# Track highest validation accuracies
		if clf == gs_xgb_clf:
			xgb_val_acc = val_acc
		elif clf == gs_cat_clf:
			cat_val_acc = val_acc
		elif clf == gs_light_clf:
			light_val_acc = val_acc
		else:
			ens_val_acc = val_acc

	# Export Data For Analysis
	xgb_results = pd.DataFrame(gs_xgb_clf.cv_results_)
	xgb_results.insert(0, 'Model', 'XGBoost')

	cat_results = pd.DataFrame(gs_cat_clf.cv_results_)
	cat_results.insert(0, 'Model', 'CatBoost')

	light_results = pd.DataFrame(gs_light_clf.cv_results_)
	light_results.insert(0, 'Model', 'LightGBM')

	# Export to CSV
	all_grid_results = pd.concat([xgb_results, cat_results, light_results], ignore_index=True)
	all_grid_results.to_csv('optimization.csv', index=False)
else:
	for clf in [xgb_clf, cat_clf, light_clf, eclf]:
		clf.fit(X_train, Y_train)

		if clf == xgb_clf:
			print("\nXGB Classifier")
		elif clf == cat_clf:
			print("\nCatBoost Classifier")
		elif clf == light_clf:
			print("\nLightGBM Classifier")
		else:
			print("\nEnsemble Classifier")

		# Train Preds and Acc
		train_pred = clf.predict(X_train)
		train_acc = clf.score(X_train, Y_train)
		print("Train Accuracy: ", train_acc)

		# Validation Preds and Acc
		val_acc = clf.score(X_val, Y_val)
		print("Validation Accuracy: ", val_acc)


# Recombine Training and Validation Data
X_train = pd.concat([X_train, X_val])
Y_train = pd.concat([Y_train, Y_val])

# best_val_acc = max([xgb_val_acc, cat_val_acc, light_val_acc, ens_val_acc])
# opt_thresh = 0
#
# # Select Best Model
# if best_val_acc == xgb_val_acc:
# 	clf = rs_xgb_clf.best_estimator_
# 	opt_thresh = xgb_opt_thresh
# elif best_val_acc == cat_val_acc:
# 	clf = rs_cat_clf.best_estimator_
# 	opt_thresh = cat_opt_thresh
# elif best_val_acc == light_val_acc:
# 	clf = rs_light_clf.best_estimator_
# 	opt_thresh = light_opt_thresh
# elif best_val_acc == ens_val_acc:
# 	clf = eclf
# 	opt_thresh = ens_opt_thresh

light_clf.fit(X_train, Y_train)

# Final Subission Preds
submission_preds = light_clf.predict(X_test)

# Export Submission Data
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
