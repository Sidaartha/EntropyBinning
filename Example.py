import numpy as np
import pandas as pd
import Entropy_Binning as eb

# Removing missing value replacements
def val_to_na(df_data, target_label=None, ignore_col=None):
	df_data_raw = df_data.copy()
	list_col = list(df_data_raw.columns.values)
	if class_label:
		if class_label in df_data.columns:
			list_col.remove(class_label)
	if ignore_col : 
		if not isinstance(ignore_col, list):
			raise AttributeError('Input ignore_col should be a list')
		if ignore_col and len(ignore_col)==0:
			raise AttributeError('Input ignore_col is empty')
		list_col = list(set(list_col).difference(set(ignore_col)))
	for i in list_col:
		nan_replacement = list(df_map.loc[df_map['mapped'] == i, 'nan_replacements'])[0]
		df_data_raw[i] = df_data_raw[i].replace(nan_replacement, np.nan)
	return df_data_raw

# Reading Data
df_train = pd.read_csv('historical-20180831_183744.csv')
df_test = pd.read_csv('historical-20180831_183744_fwd.csv')
df_map = pd.read_csv('historical-20180831_183744.featuremapping.csv')

# Target column
class_label='r' 

# Removing missing value replacements
df_train_raw = val_to_na(df_train, target_label = class_label)
df_test_raw = val_to_na(df_test, target_label = class_label)

# Droping duplicate rows from train
df_train_raw = df_train_raw.drop_duplicates()

# All continuous feature columns
features=['f986', 'f988', 'f989', 'f994', 'f998', 'f1001', 'f1002', 'f1010', 'f1003', 'f1004', 'f1006', 'f1009', 'f1052', 'f1053', 'f778', 'f1054', 'f534', 'f1149', 'f1092', 'f1055', 'f1056', 'f1057', 'f1093', 'f1094', 'f1058', 'f1061', 'f1062', 'f1122', 'f1123', 'f1124', 'f1099', 'f1100', 'f1104', 'f1065', 'f1066', 'f1067', 'f1074', 'f1075', 'f1076', 'f1077', 'f1079', 'f1080', 'f635']
features=['f1052']

# Discretization initializer
dis = eb.MDLP_Discretizer(dataset=df_train_raw, class_label=class_label, features=features, min_bins=8, min_freq=2)

print(dis.bins_range())
#### Output : Range of each bin left-inclusive [a, b)
#### {'f1124': [-inf, 13500.0, 16500.0, 24500.0, 43500.0, 49500.0, 50500.0, 75500.0, inf], 
####   'f998': [-inf, 0.5, 1.5, 2.5, 3.5, 4.5, 7.5, inf], .......}

print(dis.bins_frequency())
#### Output : Total frequency of each bin
#### {'f1124': [1529, 952, 2019, 2859, 229, 230, 331, 239], 
####   f998': [4923, 1871, 662, 352, 187, 187, 206], .......}

for i in np.sort(df_train_raw[class_label].unique()):
	print(dis.bins_frequency_target(target = i))
#### Output : Each class frequency of each bin 
#### For class '-1', i.e, Non defaulters
#### {'f1124': [584, 442, 1039, 1589, 169, 120, 216, 129], 
####   'f998': [2143, 1026, 442, 227, 132, 157, 161], .......}
#### For class '+1', i.e, Defaulters
#### {'f1124': [945, 510, 980, 1270, 60, 110, 115, 110], 
####   'f998': [2780, 845, 220, 125, 55, 30, 45], .......}

# Similarly for test set( FWD data )
print(dis.bins_frequency_fwd(data=df_test_raw))
if class_label in df_test_raw:
	for i in np.sort(df_test_raw[class_label].unique()):
		print(dis.bins_frequency_fwd_target(data=df_test_raw, target = i))