# General
from data import *

# Sklearn
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Other Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Decision TRee Models
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
	'n_estimators': 100,
	'max_depth': 3,
	'random_state': RANDOM_SEED
}

rf_param_grid = {
	'n_estimators': [125],
	'max_depth': [6],
	'min_samples_split': [2],
	'min_samples_leaf': [3],
	'max_features': ['sqrt'],
	'bootstrap': [False, True]
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
	'n_estimators': [118, 116, 117],
	'max_depth': [7],
	'min_samples_split': [5],
	'min_samples_leaf': [2],
	'max_features': ['sqrt'],
	'bootstrap': [False]
}

et_clf = ExtraTreesClassifier(**et_params)

et_clf = GridSearchCV(
	estimator=et_clf,
	param_grid=et_param_grid,
	cv=5,
	scoring='accuracy'
)

# Geometric Models

# SVC
svc_params = {
	'C': 9,
	'kernel': 'rbf',
	'gamma': 'scale',
	'probability': False,
	'random_state': RANDOM_SEED
}

svc_param_grid = {
	'C': [9],
	'kernel': ['rbf'],
	'gamma': ['scale'],
	'probability': [False],
	'random_state': [RANDOM_SEED]
}

svc_clf = SVC(**svc_params)

# [1.6, 0.4, 0.4, 0.3, 1.5]

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
	'n_neighbors': [6],
	'weights': ['uniform'],
	'algorithm': ['auto'],
	'leaf_size': [20],
	'p': [1],
	'n_jobs': [-1]
}

# Cross Validation Accuracy:  0.8271797125102003
# Train Accuracy:  0.8585858585858586

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
    ('et', et_clf)
]

RS_weights = [1.9, 0.6, 1.4, 1.8, 1.6]

# Ensembles
v_clf = VotingClassifier(
	estimators=base_models,
	voting='hard',
	weights=RS_weights
)

s_clf = StackingClassifier(
	estimators=base_models,
	final_estimator=LogisticRegression(),
	cv=2,
	n_jobs=-1
)

def test_models(models = [], ensembles = []):
	cv_acc = 0
	train_acc = 0
	val_acc = 0

	for model in models:
		print("Model: ", model[0])
		model[1].fit(X_train_CV, Y_train_CV)

		# Train Acc
		train_acc = model[1].score(X_train_DT, Y_train)
		print("Train Accuracy: ", train_acc)

		# Validation Acc
		val_acc = model[1].score(X_val, Y_val)
		print("Validation Accuracy: ", val_acc, '\n')

	for model in ensembles:
		model.fit(X_train_CV, Y_train_CV)

		# Cross-Validation Acc
		cv_acc = model.score(X_train_CV, Y_train_CV)
		print("Cross Validation Accuracy: ", cv_acc)

		# Train Acc
		train_acc = model.score(X_train_DT, Y_train)
		print("Train Accuracy: ", train_acc)

		# Validation Acc
		val_acc = model.score(X_val, Y_val)
		print("Validation Accuracy: ", val_acc, '\n')

	return cv_acc, train_acc, val_acc

# test_models(base_models, [v_clf, s_clf])

def test_model_weights(ensemble, n_iter):
	best_agg_acc = 0
	best_weights = []

	possible_weights = [i / 10 for i in range(1, 21)]

	for combo in range(n_iter):
		weights = [random.choice(possible_weights) for _ in range(len(ensemble.estimators))]

		clf = VotingClassifier(
			estimators=base_models,
			voting='soft',
			weights=weights,
			n_jobs=-1
		)

		print("Weights: ", weights)

		cv_acc, test_acc, val_acc = test_models(ensembles=[clf])

		if sum([cv_acc, test_acc, val_acc]) / 3 > best_agg_acc:
			best_weights = weights

	print("Optimal Weights: ", best_weights)
	return best_weights

# opt_weights = test_model_weights(v_clf, 5000)
