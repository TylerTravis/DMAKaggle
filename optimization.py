import pandas as pd
import ast
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# Session 3:
	# Train Accuracy: 0.8383838383838383
	# Cross-Validation Accuracy: 0.8215617349821104

# Session 4:
	# Layers: 3-6 -> 3-5
	# Nodes: 10-50 -> 21-42
	# Removed Adam
	# Removed Tanh
	# Alpha Range: 0.001-0.5 -> 0.001-0.2
	# Train Accuracy: 0.8451178451178452
	# Cross-Validation Accuracy: 0.8159625886636117

# Session 5:
	# Layers: 3-5 -> 3-6
	# Alpha Range: 0.001-0.2 -> 0.001-0.015
	# Train Accuracy: 0.8574635241301908
	# Cross-Validation Accuracy: 0.8170610758897746

# Session 6:
	# Nodes: 32 for all
	# Layers: 3-5 -> 3
	# Train Accuracy: 0.8529741863075196
	# Cross-Validation Accuracy: 0.8125729709371667

# Session 7:
	# Nodes: 32 -> 28-38
	# Layers: 3 -> 4
	# Solver: Added Adam
	# Activation: Added Tanh
	# Train Accuracy: 0.8597081930415263
	# Cross-Validation Accuracy: 0.8170610758897746
	# BEST HYPERPARAMETERS: {'solver': 'lbfgs', 'hidden_layer_sizes': (29, 34, 35, 35), 'alpha': 0.005, 'activation': 'tanh'}

# Session 8:
	# Layers: 4 -> 4-5
	# Solver: Removed Adam
	# Activation: Removed Tanh
	# Train Accuracy:  0.8664421997755332
	# Cross-Validation Accuracy:  0.8193208210407381
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (33, 31, 28, 28), 'alpha': 0.01, 'activation': 'relu'}

# Session 9:
	# Correlation Threshold: 0.2 -> 0.3
	# Nodes 28-38 -> 31-33
	# Train Accuracy:  0.8552188552188552
	# Cross Validation Accuracy:  0.8170422446801832
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (33, 31, 31, 33, 33), 'alpha': 0.001, 'activation': 'relu'}

# Session 10:
	# Correlation Threshold: 0.3 -> 0.25
	# Train Accuracy: 0.8597081930415263
	# Cross-Validation Accuracy: 0.8215868432615654
	# BEST HYPERPARAMETERS: 'solver': 'lbfgs', 'hidden_layer_sizes': (31, 31, 32, 32, 31), 'alpha': 0.01, 'activation': 'relu'

# Session 11:
	# !! Titles Inclusion !!
	# Correlation Threshold DROPPED
	# Layers: 4 -> 2-3
	# Nodes 31-33 -> 20-40
	# Train Accuracy:  0.9663299663299664
	# Cross-Validation Accuracy:  0.8035967610319503
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (24, 29, 26), 'alpha': 0.01, 'activation': 'relu'}

# Session 12:
	# Layers: 2-3 -> 2
	# Train Accuracy: 0.9461279461279462
	# Cross-Validation Accuracy: 0.8025045508756513
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (38, 28), 'alpha': 0.01, 'activation': 'relu'}
	# Note: For seed range 20-40, mean 28 appears for both layers
	# Alpha = 0.1 appears optimal

# Session 13:
	# Solver: Added Adam
	# Activation: Added Tanh
	# Nodes 20-40 -> 10-47
	# Train Accuracy: 0.9046015712682379
	# Cross-Validation Accuracy: 0.8294331805913
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (38, 28), 'alpha': 0.01, 'activation': 'relu'}
	# Note: For seed range 20-37, we have mean 21 and 23
	# Alpha = 0.1 continues to appear optimal

# Session 14:
	# Solver: !Removed LBFGS!
	# Activation: !Removed ReLU!
	# Alpha: 0.001, 0.005, 0.01 -> 0.005, 0.01, 0.015
	# Nodes: 10-47 -> 19-25
	# Train Accuracy: 0.8866442199775533
	# Cross-Validation Accuracy: 0.8249387985688281
	# BEST HYPERPARAMETERS:  {'solver': 'adam', 'hidden_layer_sizes': (25, 23), 'alpha': 0.01, 'activation': 'tanh'}

# Session 15:
	# Solver: Removed Adam, Added LBFGS
	# Activation: Removed Tanh, Added ReLU
	# Train Accuracy:  0.9506172839506173
	# Cross-Validation Accuracy:  0.80585022911305
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (25, 25), 'alpha': 0.05, 'activation': 'relu'}

# Session 16:
	# Alpha: 0.005, 0.01, 0.015 -> 0.05, 0.075, 0.01
	# Train Accuracy:  0.941638608305275
	# Cross-Validation Accuracy:  0.8047329106772958
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (20, 22), 'alpha': 0.05, 'activation': 'relu'}
	# Note: Alpha = 0.075 appears optimal

# Session 17:
	# Layers: 2 -> 2-3
	# Train Accuracy:  0.9584736251402918
	# Cross-Validation Accuracy:  0.8036155922415416
	# BEST HYPERPARAMETERS:  {'solver': 'lbfgs', 'hidden_layer_sizes': (23, 22, 19), 'alpha': 0.075, 'activation': 'relu'}
	# Note: Alpha = 0.075 appears optimal again

# Model Change: MLP -> HGB

# Session 18
	# Learning Rate: 0.01-0.1
	# Max Iter: 50-500
	# Max Depth: 1-10
	# L2: 0.1-1.0
	# Train Accuracy:  0.9113355780022446
	# Cross-Validation Accuracy:  0.8474044316113238
	# BEST HYPERPARAMETERS:  {'max_iter': 300, 'max_depth': 6, 'learning_rate': 0.05, 'l2_regularization': 0.1}

# Session 19
	# Learning Rate: 0.005-0.05
	# Max Iter: 150-450
	# Max Depth: 5-15
	# L2: 0.1-1.0
	# BEST HYPERPARAMETERS:  {'max_iter': 300, 'max_depth': 6, 'learning_rate': 0.05, 'l2_regularization': 0.1}
	# Train Accuracy:  0.941638608305275
	# Cross-Validation Accuracy:  0.848521750047078
	# BEST HYPERPARAMETERS:  {'max_iter': 350, 'max_depth': 9, 'learning_rate': 0.025, 'l2_regularization': 0.6000000000000001}

# Session 20
	# Learning Rate: 0.01-0.04
	# Max Iter: 200-500
	# Max Depth: 5-15
	# L2: 0.1-1.0
	# Train Accuracy:  0.9079685746352413
	# Cross-Validation Accuracy:  0.8507375557089951
	# BEST HYPERPARAMETERS:  {'max_iter': 350, 'max_depth': 15, 'learning_rate': 0.015, 'l2_regularization': 0.5}

# Session 21
	# Learning Rate: 0.015-0.03
	# Max Iter: 200-450
	# Max Depth: 10-20
	# L2: 0.1-0.5
	# Train Accuracy:  0.9090909090909091
	# Cross-Validation Accuracy:  0.8518674282844767
	# BEST HYPERPARAMETERS:  {'max_iter': 300, 'max_depth': 17, 'learning_rate': 0.015, 'l2_regularization': 0.1}

# Session 22
	# Learning Rate: 0.005-0.03
	# Max Iter: 200-350
	# Max Depth: 11-16
	# L2: 0.1-0.4
	# Train Accuracy:  0.9158249158249159
	# Cross-Validation Accuracy:  0.8518737053543406
	# BEST HYPERPARAMETERS:  {'max_iter': 250, 'max_depth': 15, 'learning_rate': 0.02, 'l2_regularization': 0.25}

# Instance Test
	# LR: 0.02
	# MI: 300
	# MD: 14
	# L2: 0.15
	# Train Accuracy:  0.9158249158249159 (not a typo)

# Instance Test
	# LR: 0.02
	# MI: 300
	# MD: 14
	# L2: 0.2
	# Train Accuracy: 0.9191919191919192

# Instance Test
	# LR: 0.02
	# MI: 300
	# MD: 13
	# L2: 0.2
	# Train Accuracy: 0.9135802469135802

# Instance Test (No Longer Imputing NaN)
	# LR: 0.02
	# MI: 300
	# MD: 13
	# L2: 0.2
	# Train Accuracy: 0.9191919191919192

# Session 23
	# Learning Rate: 0.005-0.03
	# Max Iter: 200-350
	# Max Depth: 11-16
	# L2: 0.1-0.5
	# Train Accuracy:  0.9438832772166106
	# Cross-Validation Accuracy:  0.8462557278262507
	# BEST HYPERPARAMETERS:  {'max_iter': 200, 'max_depth': 14, 'learning_rate': 0.025, 'l2_regularization': 0.15}

# Session 24
	# Learning Rate: 0.02-0.04
	# Max Iter: 200-400
	# Max Depth: 9-15
	# L2: 0.1-0.5
	# Train Accuracy:  0.9506172839506173
	# Cross-Validation Accuracy:  0.8485029188374866
	# BEST HYPERPARAMETERS:  {'max_iter': 200, 'max_depth': 12, 'learning_rate': 0.04, 'l2_regularization': 0.4}

# Model Change: HGB -> XGB

# Session 25

# Data
opt_df = pd.read_csv('/Users/johnd/Desktop/Berkeley/DMA/kaggle/optimization.csv')

opt_filtered = opt_df.nsmallest(100, 'rank_test_score')
opt_sorted = opt_filtered.sort_values(by='rank_test_score', ascending=False)

# Best Parameters
best_h = opt_sorted['params'].iloc[0]
print("BEST HYPERPARAMETERS: ", best_h)

columns_to_analyze = [
	'param_subsample',
	'param_reg_lambda',
	'param_reg_alpha',
	'param_n_estimators',
	'param_min_child_weight',
	'param_max_depth',
	'param_learning_rate',
	'param_gamma',
	'param_colsample_bytree'
]

for col in columns_to_analyze:
	proportions = opt_sorted[col].value_counts(normalize=True) * 100
	print(proportions, '%')

	plt.hist(opt_sorted[col], bins=10, color='red', edgecolor='black')

	plt.title(col)
	plt.xlabel('Value')
	plt.ylabel('Count')
	plt.show()