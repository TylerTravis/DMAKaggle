# General
import kagglehub
import pandas as pd
import numpy as np
import random

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Other Models
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
	df['Title'] = df['Title'].replace(['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Mlle', 'Ms', 'Mme', 'Mrs'], 'Rare')

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
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
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
		['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Mrs', 'Miss', 'Mlle', 'Ms'], 'Rare')
	# Embarked
	df['Embarked'] = df['Embarked'].astype('category').cat.codes

	# Cabin
	# df['Cabin'] = df['Cabin'].str.extract(r'([A-Za-z])', expand=False).astype('category').cat.codes

	# Added Features
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['IsChild'] = df['Age'].apply(lambda x: int(x < 7))
	df['IsYouth'] = df['Age'].apply(lambda x: int(7 <= x < 14))
	df['HasCabin'] = df['Cabin'].notna().astype(int)
	df['RareTitle'] = df['Title'].apply(lambda x: int(x == 'Rare'))
	df['ExpensiveFare'] = df['Fare'].apply(lambda x: int(x > 39))
	df['IsChildAndFamilyMemberSurvived'] = df.apply(lambda row: int(row['IsChild'] and row['GroupSurvivalRate'] > 0), axis=1)

	return df

# Decision Tree Data
X_train_DT = process_data_for_decision_tree(train_df)
X_test_DT = process_data_for_decision_tree(test_df)

cols_to_drop_DT = ['Name', 'Title', 'Cabin', 'Ticket', 'Fare', 'FamilySize']

# Test & Train
X_train_DT = X_train_DT.drop(columns=['Survived', 'PassengerId'] + cols_to_drop_DT)
Y_train_DT = train_df['Survived']

X_test_DT = X_test_DT.drop(columns=['PassengerId'] + cols_to_drop_DT)
X_test_DT = X_test_DT.reindex(columns=X_train_DT.columns, fill_value=0)

X_train_DT = X_train_DT.add_suffix('_DT')
X_test_DT = X_test_DT.add_suffix('_DT')

# Geometric Data
X_train_G = process_data(train_df)
X_test_G = process_data(test_df)

cols_to_drop_G = ['Name', 'Title', 'Cabin', 'Embarked', 'Pclass', 'Ticket', 'Fare', 'Age']

# Test & Train
X_train_G = X_train_G.drop(columns=['Survived', 'PassengerId'] + cols_to_drop_G)
Y_train_G = train_df['Survived']

X_test_G = X_test_G.drop(columns=['PassengerId'] + cols_to_drop_G)
X_test_G = X_test_G.reindex(columns=X_train_G.columns, fill_value=0)

X_train_G = X_train_G.add_suffix('_G')
X_test_G = X_test_G.add_suffix('_G')

# Merge DataFrames
X_train_ALL = pd.concat([X_train_DT, X_train_G], axis=1)
X_test_ALL = pd.concat([X_test_DT, X_test_G], axis=1)

# Validation Split
X_train_ALL, X_val_ALL, Y_train, Y_val = train_test_split(
    X_train_ALL, Y_train_DT, test_size=0.15, random_state=RANDOM_SEED
)

# Feature Lists
dt_features = X_train_DT.columns.tolist()
g_features = X_train_G.columns.tolist()

# BASE MODELS

# Gradient Boosting Params
gb_params = {
	'max_depth': 3,
	'learning_rate': 0.05,
	'n_estimators': 70,
	'min_child_weight': 4,
	'subsample': 0.6,
	'colsample_bytree': 0.75,
	'l1_leaf_reg': 0,
	'l2_leaf_reg': 0,
	'random_state': RANDOM_SEED,
	'eval_metric': 'logloss'
}

gb_param_grid = {
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
	'max_depth': gb_params['max_depth'],
	'learning_rate': gb_params['learning_rate'],
	'n_estimators': gb_params['n_estimators'],
	'min_child_weight': gb_params['min_child_weight'],
	'subsample': gb_params['subsample'],
	'colsample_bytree': gb_params['colsample_bytree'],
	'gamma': 0.175,
	'reg_alpha': gb_params['l1_leaf_reg'],
	'reg_lambda': gb_params['l2_leaf_reg'],
	'random_state': gb_params['random_state'],
	'eval_metric': gb_params['eval_metric'],
}

xgb_param_grid = {
	'max_depth': gb_param_grid['max_depth'],
	'learning_rate': gb_param_grid['learning_rate'],
	'n_estimators': gb_param_grid['n_estimators'],
	'min_child_weight': gb_param_grid['min_child_weight'],
	'subsample': gb_param_grid['subsample'],
	'colsample_bytree': gb_param_grid['subsample'],
	'gamma': [0, 0.1, 0.25, 0.5],
	'reg_alpha': gb_param_grid['l1_leaf_reg'],
	'reg_lambda': gb_param_grid['l2_leaf_reg']
}

xgb_clf = xgb.XGBClassifier(**xgb_params)

# CatBoost Classifier
cat_params = {
	'objective': 'Logloss',
	'random_state': gb_params['random_state'],
	'eval_metric': 'Accuracy',
	'iterations': gb_params['n_estimators'],
	'min_data_in_leaf': gb_params['min_child_weight'],
	'learning_rate': gb_params['learning_rate'],
	'l2_leaf_reg': gb_params['l2_leaf_reg'],
	'subsample': gb_params['subsample'],
	'depth': gb_params['max_depth'],
	'verbose': 0,
	'od_type': 'Iter'
}

cat_param_grid = {
	'objective': ['Logloss'],
	'eval_metric': ['Accuracy'],
	'depth': gb_param_grid['max_depth'],
	'learning_rate': gb_param_grid['learning_rate'],
	'iterations': gb_param_grid['n_estimators'],
	'min_data_in_leaf': gb_param_grid['min_child_weight'],
	'subsample':gb_param_grid['subsample'],
	'l2_leaf_reg': gb_param_grid['l2_leaf_reg'],
	'random_state': [RANDOM_SEED],
}

cat_clf = cb.CatBoostClassifier(**cat_params)

# LightGBM Classifier
light_params = {
	'task': 'predict',
	'boosting': 'gbdt',
	'data_sample_strategy': 'bagging',
	'objective': 'binary',
	'lambda_l1': 0,
	'lambda_l2': 0,
	'max_depth': gb_params['max_depth'],
	'min_gain_to_split': 0.3,
	'n_estimators': gb_params['n_estimators'],
	'max_bin': 30,
	'learning_rate': gb_params['learning_rate'],
	'verbose': -1,
	'random_state': gb_params['random_state']
}

light_param_grid = {
	'max_depth': gb_param_grid['max_depth'],
	'min_gain_to_split': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
	'n_estimators': gb_param_grid['n_estimators'],
	'max_bin': [10, 20, 30, 40, 50],
	'learning_rate':gb_param_grid['learning_rate'],
	'random_state': [RANDOM_SEED]
}

light_clf = lgb.LGBMClassifier(**light_params)

# Random Forest
rf_params = {
	'n_estimators': 125,
	'max_depth': 6,
	'min_samples_split': 2,
	'min_samples_leaf': 3,
	'max_features': 'sqrt',
	'bootstrap': False
}

rf_param_grid = {
	'n_estimators': [85, 100, 115],
	'max_depth': [3],
	'min_samples_split': [4, 5, 6],
	'min_samples_leaf': [1, 2, 3, 4],
	'max_features': ['sqrt', 'log2'],
	'bootstrap': [True, False]
}

rf_clf = RandomForestClassifier(**rf_params)

rf_clf = GridSearchCV(
	estimator=rf_clf,
	param_grid=rf_param_grid,
	cv=5,
	scoring='accuracy'
)

# Extra Trees
et_params = {
	'n_estimators': 118,
	'max_depth': 7,
	'min_samples_split': 5,
	'min_samples_leaf': 2,
	'max_features': 'sqrt',
	'bootstrap': False
}

et_param_grid = {
	'n_estimators': [85, 100, 115],
	'max_depth': [3],
	'min_samples_split': [4, 5, 6],
	'min_samples_leaf': [1, 2, 3, 4],
	'max_features': ['sqrt', 'log2'],
	'bootstrap': [True, False]
}

et_clf = ExtraTreesClassifier(**et_params)

et_clf = GridSearchCV(
	estimator=et_clf,
	param_grid=et_param_grid,
	cv=5,
	scoring='accuracy'
)

# SVC
svc_params = {
	'C': 9,
	'kernel': 'rbf',
	'gamma': 'scale',
	'probability': False,
	'random_state': RANDOM_SEED
}

svc_param_grid = {
	'C': [0.1, 1, 5, 10, 25, 40, 60, 80],
	'kernel': ['linear', 'rbf'],
	'gamma': ['scale', 'auto'],
	'probability': [True],
	'random_state': [RANDOM_SEED]
}

svc_clf = SVC(**svc_params)

svc_clf = GridSearchCV(
	estimator=svc_clf,
	param_grid=svc_param_grid,
	cv=5,
	scoring='accuracy'
)

# K Nearest Neighbors
knn_params = {
	'n_neighbors': 6,
	'weights': 'uniform',
	'algorithm': 'auto',
	'leaf_size': 20,
	'p': 1,
	'n_jobs': -1
}

knn_param_grid = {
	'n_neighbors': [3, 5, 7],
	'weights': ['uniform', 'distance'],
	'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	'leaf_size': [20, 25, 30, 35, 40],
	'p': [1, 2],
	'n_jobs': [-1]
}

knn_clf = KNeighborsClassifier(**knn_params)

knn_clf = GridSearchCV(
	estimator=knn_clf,
	param_grid=knn_param_grid,
	cv=5,
	scoring='accuracy'
)

base_models = [
    ('xgb', xgb_clf),
    ('cat', cat_clf),
    ('light', light_clf),
    ('rf', rf_clf),
    ('et', et_clf),
    ('svc', svc_clf),
    ('knn', knn_clf)
]

# Ensembles
v_clf = VotingClassifier(
	estimators=base_models,
	voting='soft',
	weights=[1, 1, 1, 1, 1, 1, 1]
)

s_clf = StackingClassifier(
	estimators=base_models,
	final_estimator=LogisticRegression(),
	cv=5,
	n_jobs=-1
)

# for name, clf in base_models + [('voting', v_clf), ('stacking', s_clf)]:
# 	print("\n" + name + " Classifier")
#
# 	clf.fit(X_train_ALL, Y_train)
#
# 	# Train Preds and Acc
# 	train_acc = clf.score(X_train_ALL, Y_train)
# 	print("Train Accuracy: ", train_acc)
#
# 	# Validation Preds and Acc
# 	val_acc = clf.score(X_val_ALL, Y_val)
# 	print("Validation Accuracy: ", val_acc)


# Recombine Training and Validation Data
predictions = []

prediction_weights = [
	0.993498049415,
	0.97789336801,
	1,
	0.967486377983,
	0.968773683389,
	0.964875402612,
	0.958409783936
]

# Fit and Predict Proabilities
for model in base_models:
	# Disjoint training
	if model[0] in ['xgb', 'cat', 'light', 'rf', 'et']:
		model[1].fit(X_train_DT, Y_train_DT)
		predictions.append(model[1].predict_proba(X_test_DT)[1])
	else:
		model[1].fit(X_train_G, Y_train_G)
		predictions.append(model[1].predict_proba(X_test_G)[1])

submission_preds = []

print(predictions)

for i, pred in predictions:
	weighted_pred = pred[1] * prediction_weights[i]
	submission_preds.append(1 if weighted_pred > 0.5 else 0)

# mean_pred = sum(prediction_weights) / len(prediction_weights)
#
# soft_preds = 1 if mean_pred > 0.5 else 0




# X_train_ALL = pd.concat([X_train_ALL, X_val_ALL])
# Y_train_ALL = pd.concat([Y_train, Y_val])

# s_clf.fit(X_train, Y_train_ALL)

# Final Subission Preds
# submission_preds = s_clf.predict(X_test_ALL)

# Export Submission Data
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
