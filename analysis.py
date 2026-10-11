from data import *
from pprint import pprint

def EDA(df, verbose = True):
	df = df.copy()
	df['Survived'] = train_df['Survived']
	df['Ticket'] = train_df['Ticket']

	cols_to_ignore = ['Name', 'PassengerId', 'Survived', 'Ticket']
	cols_to_analyze = [col for col in df.columns if col not in cols_to_ignore]

	highest_correlations = []

	for col in df.columns:
		if verbose:
			print(df[col].value_counts(), '\n')

		if col in cols_to_analyze:
			corr = df[col].corr(df['Survived'])
			highest_correlations.append((col, corr))

	highest_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
	highest_correlations_formatted = [(col, round(float(corr), 3)) for col, corr in highest_correlations]

	print('HIGHEST CORRELATIONS\n')
	pprint(highest_correlations_formatted)

	return highest_correlations

EDA(X_train_DT, False)