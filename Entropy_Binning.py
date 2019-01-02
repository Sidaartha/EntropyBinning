import numpy as np
import pandas as pd
from math import log

def entropy(data_classes, base=2):
	'''
	Computes the entropy of a set of labels (class instantiations)
	:param base: logarithm base for computation
	:param data_classes: Series with labels of examples in a dataset
	:return: value of entropy
	'''
	if not isinstance(data_classes, pd.core.series.Series):
		raise AttributeError('input array should be a pandas series')
	classes = data_classes.unique()
	N = len(data_classes)
	ent = 0  # initialize entropy

	# iterate over classes
	for c in classes:
		partition = data_classes[data_classes == c]  # data with class = c
		proportion = len(partition) / N
		#update entropy
		ent -= proportion * log(proportion, base)

	return ent

def cut_point_information_gain(dataset, cut_point, feature_label, class_label):
	'''
	Return de information gain obtained by splitting a numeric attribute in two according to cut_point
	:param dataset: pandas dataframe with a column for attribute values and a column for class
	:param cut_point: threshold at which to partition the numeric attribute
	:param feature_label: column label of the numeric attribute values in data
	:param class_label: column label of the array of instance classes
	:return: information gain of partition obtained by threshold cut_point
	'''
	if not isinstance(dataset, pd.core.frame.DataFrame):
		raise AttributeError('input dataset should be a pandas data frame')

	entropy_full = entropy(dataset[class_label])  # compute entropy of full dataset (w/o split)

	#split data at cut_point
	data_left = dataset[dataset[feature_label] <= cut_point]
	data_right = dataset[dataset[feature_label] > cut_point]
	(N, N_left, N_right) = (len(dataset), len(data_left), len(data_right))

	gain = entropy_full - (N_left / N) * entropy(data_left[class_label]) - \
		(N_right / N) * entropy(data_right[class_label])

	return gain

class MDLP_Discretizer(object):
	def __init__(self, dataset, class_label, features=None, min_bins=None, min_freq=None):
		'''
		initializes discretizer object:
			saves raw copy of data and creates self._data with only features to discretize and class
			computes initial entropy (before any splitting)
			self._features = features to be discretized
			self._classes = unique classes in raw_data
			self._class_name = label of class in pandas dataframe
			self._data = partition of data with only features of interest and class
			self._cuts = dictionary with cut points for each feature
		:param dataset: pandas dataframe with data to discretize
		:param class_label: name of the column containing class in input dataframe
		:param features: if !None, features that the user wants to discretize specifically
		:return:
		'''

		if not isinstance(dataset, pd.core.frame.DataFrame):  # class needs a pandas dataframe
			raise AttributeError('Input dataset should be a pandas data frame')

		self._data_raw = dataset #copy or original input data
		self._size = len(self._data_raw)

		self._class_name = class_label

		self._classes = np.sort(self._data_raw[self._class_name].unique())

		#if user specifies which attributes to discretize
		if features:
			self._features = [f for f in features if f in self._data_raw.columns]  # check if features in dataframe
			missing = set(features) - set(self._features)  # specified columns not in dataframe
			if missing:
				print ('WARNING: user-specified features %s not in input dataframe' % str(missing))
		else:  # then we need to recognize which features are numeric
			numeric_cols = self._data_raw._data.get_numeric_data().items
			self._features = [f for f in numeric_cols if f != class_label]

		self.min_dict = {}
		self.max_dict = {}
		for i in self._features:
			range_min, range_max = self._data_raw[i].min(), self._data_raw[i].max()
			self.min_dict[i] = range_min
			self.max_dict[i] = range_max
		self._partition_dict = {}
		self._candidate_dict = {}
		self._partition_freq = {}

		if min_bins and min_freq:
			self._min_bins = min_bins
			self._min_freq = min_freq

		#other features that won't be discretized
		self._ignored_features = set(self._data_raw.columns) - set(self._features)

		#create copy of data only including features to discretize and class
		self._data = self._data_raw.loc[:, self._features + [class_label]]
		#pre-compute all boundary points in dataset
		self._boundaries = self.compute_boundary_points_all_features()
		#initialize feature bins with empty arrays
		self._cuts = {f: [] for f in self._features}
		#get cuts for all features
		self.all_features_accepted_cutpoints()
		#further binning
		if min_bins and min_freq:
			self.all_features_min_criteria_cutpoints()

	def MDLPC_criterion(self, data, feature, cut_point):
		'''
		Determines whether a partition is accepted according to the MDLPC criterion
		:param feature: feature of interest
		:param cut_point: proposed cut_point
		:param partition_index: index of the sample (dataframe partition) in the interval of interest
		:return: True/False, whether to accept the partition
		'''
		#get dataframe only with desired attribute and class columns, and split by cut_point
		data_partition = data.copy(deep=True)
		data_left = data_partition[data_partition[feature] <= cut_point]
		data_right = data_partition[data_partition[feature] > cut_point]

		#compute information gain obtained when splitting data at cut_point
		cut_point_gain = cut_point_information_gain(dataset=data_partition, cut_point=cut_point,
													feature_label=feature, class_label=self._class_name)
		#compute delta term in MDLPC criterion
		N = len(data_partition) # number of examples in current partition
		partition_entropy = entropy(data_partition[self._class_name])
		k = len(data_partition[self._class_name].unique())
		k_left = len(data_left[self._class_name].unique())
		k_right = len(data_right[self._class_name].unique())
		entropy_left = entropy(data_left[self._class_name])  # entropy of partition
		entropy_right = entropy(data_right[self._class_name])
		delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

		#to split or not to split
		gain_threshold = (log(N - 1, 2) + delta) / N
		if cut_point_gain > gain_threshold:
			return True
		else:
			return False

	def feature_boundary_points(self, data, feature):
		'''
		Given an attribute, find all potential cut_points (boundary points)
		:param feature: feature of interest
		:param partition_index: indices of rows for which feature value falls whithin interval of interest
		:return: array with potential cut_points
		'''
		#get dataframe with only rows of interest, and feature and class columns
		data_partition = data.copy(deep=True)
		data_partition.sort_values(feature, ascending=True, inplace=True)

		boundary_points = []

		#add temporary columns
		data_partition['class_offset'] = data_partition[self._class_name].shift(1)  # column where first value is now second, and so forth
		data_partition['feature_offset'] = data_partition[feature].shift(1)  # column where first value is now second, and so forth
		data_partition['feature_change'] = (data_partition[feature] != data_partition['feature_offset'])
		data_partition['mid_points'] = data_partition.loc[:, [feature, 'feature_offset']].mean(axis=1)

		potential_cuts = data_partition[data_partition['feature_change'] == True].index[1:]
		sorted_index = data_partition.index.tolist()

		for row in potential_cuts:
			old_value = data_partition.loc[sorted_index[sorted_index.index(row) - 1]][feature]
			new_value = data_partition.loc[row][feature]
			old_classes = data_partition[data_partition[feature] == old_value][self._class_name].unique()
			new_classes = data_partition[data_partition[feature] == new_value][self._class_name].unique()
			if len(set.union(set(old_classes), set(new_classes))) > 1:
				boundary_points += [data_partition.loc[row]['mid_points']]

		return set(boundary_points)

	def compute_boundary_points_all_features(self):
		'''
		Computes all possible boundary points for each attribute in self._features (features to discretize)
		:return:
		'''
		boundaries = {}
		for attr in self._features:
			data_partition = self._data.loc[:, [attr, self._class_name]]
			boundaries[attr] = self.feature_boundary_points(data=data_partition, feature=attr)
		return boundaries

	def boundaries_in_partition(self, data, feature):
		'''
		From the collection of all cut points for all features, find cut points that fall within a feature-partition's
		attribute-values' range
		:param data: data partition (pandas dataframe)
		:param feature: attribute of interest
		:return: points within feature's range
		'''
		range_min, range_max = (data[feature].min(), data[feature].max())
		return set([x for x in self._boundaries[feature] if (x > range_min) and (x < range_max)])

	def best_cut_point(self, data, feature):
		'''
		Selects the best cut point for a feature in a data partition based on information gain
		:param data: data partition (pandas dataframe)
		:param feature: target attribute
		:return: value of cut point with highest information gain (if many, picks first). None if no candidates
		'''
		candidates = self.boundaries_in_partition(data=data, feature=feature)
		# candidates = self.feature_boundary_points(data=data, feature=feature)
		if not candidates:
			return None
		gains = [(cut, cut_point_information_gain(dataset=data, cut_point=cut, feature_label=feature,
												  class_label=self._class_name)) for cut in candidates]
		gains = sorted(gains, key=lambda x: x[1], reverse=True)

		return gains[0][0] #return cut point

	def best_cut_point_min_freq(self, data, feature):

		freq = self._min_freq

		candidates = self.boundaries_in_partition(data=data, feature=feature)
		# candidates = self.feature_boundary_points(data=data, feature=feature)
		if not candidates:
			return None
		gains = [(cut, cut_point_information_gain(dataset=data, cut_point=cut, feature_label=feature, class_label=self._class_name), [sum(i[0] for i in self.frequency_partition(data=data, feature=feature, cut_point=cut)), sum(i[1] for i in self.frequency_partition(data=data, feature=feature, cut_point=cut))]) for cut in candidates]
		gains = sorted(gains, key=lambda x: x[1], reverse=True)

		best_cut = None
		for i in gains:
			if i[2][0] > (self._size)*freq/100 and i[2][1] > (self._size)*freq/100:
				best_cut = i[0]
				break

		return best_cut #return cut point

	def single_feature_accepted_cutpoints(self, feature, partition_index=pd.DataFrame().index):
		'''
		Computes the cuts for binning a feature according to the MDLP criterion
		:param feature: attribute of interest
		:param partition_index: index of examples in data partition for which cuts are required
		:return: list of cuts for binning feature in partition covered by partition_index
		'''
		if partition_index.size == 0:
			partition_index = self._data.index  # if not specified, full sample to be considered for partition

		data_partition = self._data.loc[partition_index, [feature, self._class_name]]

		#exclude missing data:
		if data_partition[feature].isnull().values.any:
			data_partition = data_partition[~data_partition[feature].isnull()]

		#stop if constant or null feature values
		if len(data_partition[feature].unique()) < 2:
			return
		#determine whether to cut and where
		cut_candidate = self.best_cut_point(data=data_partition, feature=feature)
		if cut_candidate == None:
			return

		decision = self.MDLPC_criterion(data=data_partition, feature=feature, cut_point=cut_candidate)
		#apply decision
		if not decision:
			cut_candidate = self.best_cut_point_min_freq(data=data_partition, feature=feature)
			if feature in self._partition_dict:
				if cut_candidate != None:
					self._partition_dict[feature].append(data_partition.index)
					self._candidate_dict[feature].append(cut_candidate)
					self._partition_freq[feature].append(self.frequency_partition(data=data_partition, feature=feature, cut_point=cut_candidate))
			else :
				if cut_candidate != None:
					self._partition_dict[feature] = [data_partition.index]
					self._candidate_dict[feature] = [cut_candidate]
					self._partition_freq[feature] = [self.frequency_partition(data=data_partition, feature=feature, cut_point=cut_candidate)]
			return  # if partition wasn't accepted, there's nothing else to do
		if decision:
			# try:
			#now we have two new partitions that need to be examined
			left_partition = data_partition[data_partition[feature] <= cut_candidate]
			right_partition = data_partition[data_partition[feature] > cut_candidate]
			if left_partition.empty or right_partition.empty:
				return #extreme point selected, don't partition

			self._cuts[feature] += [cut_candidate]  # accept partition
			self.single_feature_accepted_cutpoints(feature=feature, partition_index=left_partition.index)
			self.single_feature_accepted_cutpoints(feature=feature, partition_index=right_partition.index)
			#order cutpoints in ascending order
			self._cuts[feature] = sorted(self._cuts[feature])

			return

	def all_features_accepted_cutpoints(self):
		'''
		Computes cut points for all numeric features (the ones in self._features)
		:return:
		'''
		for attr in self._features:
			self.single_feature_accepted_cutpoints(feature=attr)
		return

	def single_features_min_criteria_cutpoints(self, feature, bins, partition_index=pd.DataFrame().index):

		all_data_partitions = self._partition_dict[feature]
		all_data_frequency  = self._partition_freq[feature]
		all_data_candidates = self._candidate_dict[feature]

		freq = self._min_freq

		for i in range(bins):
			if len(all_data_partitions) == 0:
				break
			gain_list = []
			for partition_index, cut_candidate, partition_freq in zip(all_data_partitions, all_data_candidates, all_data_frequency):
				data_partition = self._data.loc[partition_index, [feature, self._class_name]]
				#exclude missing data:
				if data_partition[feature].isnull().values.any:
					data_partition = data_partition[~data_partition[feature].isnull()]
				#stop if constant or null feature values
				if len(data_partition[feature].unique()) < 2:
					gain_list.append(0)

				else : 
					gain_list.append(cut_point_information_gain(dataset=data_partition, cut_point=cut_candidate, feature_label=feature, class_label=self._class_name))
			
			if max(gain_list) != 0:
				max_gain = max(gain_list)
				max_gain_index = gain_list.index(max_gain)

				cut_candidate_chosen = all_data_candidates[max_gain_index]
				data_partitions_chosen = all_data_partitions[max_gain_index]
				data_partitions_chosen = self._data.loc[data_partitions_chosen, [feature, self._class_name]]

				#now we have two new partitions that need to be examined
				left_partition = data_partitions_chosen[data_partitions_chosen[feature] <= cut_candidate_chosen]
				left_cut_candidate = self.best_cut_point_min_freq(data=left_partition, feature=feature)
				right_partition = data_partitions_chosen[data_partitions_chosen[feature] > cut_candidate_chosen]
				right_cut_candidate = self.best_cut_point_min_freq(data=right_partition, feature=feature)

				# all_data_partitions.remove(all_data_partitions[max_gain_index])
				del all_data_partitions[max_gain_index]
				all_data_candidates.remove(all_data_candidates[max_gain_index])
				all_data_frequency.remove(all_data_frequency[max_gain_index])

				if left_cut_candidate != None:
					all_data_partitions.append(left_partition.index)
					all_data_candidates.append(left_cut_candidate)
					all_data_frequency.append(self.frequency_partition(data=left_partition, feature=feature, cut_point=left_cut_candidate))
				if right_cut_candidate != None:
					all_data_partitions.append(right_partition.index)
					all_data_candidates.append(right_cut_candidate)
					all_data_frequency.append(self.frequency_partition(data=right_partition, feature=feature, cut_point=right_cut_candidate))

				self._cuts[feature] += [cut_candidate_chosen]

			#order cutpoints in ascending order
			self._cuts[feature] = sorted(self._cuts[feature])

		return

	def all_features_min_criteria_cutpoints(self):

		for attr in self._features:
			if len(self._cuts[attr])+1 < self._min_bins:
				bins_required = self._min_bins - (len(self._cuts[attr])+1)
				self.single_features_min_criteria_cutpoints(feature=attr, bins=bins_required)
		return

	def frequency_partition(self, data, feature, cut_point):
		data_left = data[data[feature] <= cut_point]
		data_right = data[data[feature] > cut_point]
		classes = data[self._class_name].unique()
		freq_list = []
		for c in classes:
			data_classes_l = data_left[self._class_name]
			data_classes_r = data_right[self._class_name]
			# list of single class on left and right
			freq_list.append([len(data_classes_l[data_classes_l == c]), len(data_classes_r[data_classes_r == c])])
		return freq_list

	def bins_range(self):
		range_dict = {}
		for attr in self._features:
			# Range [a, b). i.e, left-inclusive
			range_dict[attr] = [-np.inf] + self._cuts[attr] + [np.inf]
		return range_dict

	def bins_frequency(self):
		freq_dict = {}
		for attr in self._features:
			range_attr = [-np.inf] + self._cuts[attr] + [np.inf]
			freq_dict[attr] = []
			for i in range(len(range_attr)-1):
				freq_dict[attr].append(len(self._data_raw.query(str(range_attr[i])+' <= '+attr+' < '+str(range_attr[i+1]))))
		return freq_dict

	def bins_frequency_target(self, target):
		freq_dict = {}
		for attr in self._features:
			range_attr = [-np.inf] + self._cuts[attr] + [np.inf]
			freq_dict[attr] = []
			for i in range(len(range_attr)-1):
				df_dis = self._data_raw.query(str(range_attr[i])+' <= '+attr+' < '+str(range_attr[i+1]))
				freq_dict[attr].append(len(df_dis[df_dis[self._class_name] == target]))
		return freq_dict

	def bins_frequency_fwd(self, data):

		if not isinstance(data, pd.core.frame.DataFrame):  # class needs a pandas dataframe
			raise AttributeError('Input dataset should be a pandas data frame')
		freq_dict = {}
		for attr in self._features:
			# Raise a warning
			if attr not in data:
				print('WARNING : '+attr+' column is missing in fwd data!')
			else:
				range_attr = [-np.inf] + self._cuts[attr] + [np.inf]
				freq_dict[attr] = []

				if data[attr].isnull().values.any:
					data = data[~data[attr].isnull()]

				for i in range(len(range_attr)-1):
					freq_dict[attr].append(len(data.query(str(range_attr[i])+' <= '+attr+' < '+str(range_attr[i+1]))))
		return freq_dict

	def bins_frequency_fwd_target(self, data, target):

		if not isinstance(data, pd.core.frame.DataFrame):  # class needs a pandas dataframe
			raise AttributeError('Input dataset should be a pandas data frame')
		freq_dict = {}
		for attr in self._features:
			# Raise a warning
			if attr not in data:
				print('WARNING : '+attr+' column is missing in fwd data!')
			else:
				range_attr = [-np.inf] + self._cuts[attr] + [np.inf]
				freq_dict[attr] = []

				if data[attr].isnull().values.any:
					data = data[~data[attr].isnull()]

				for i in range(len(range_attr)-1):
					df_dis = data.query(str(range_attr[i])+' <= '+attr+' < '+str(range_attr[i+1]))
					freq_dict[attr].append(len(df_dis[df_dis[self._class_name] == target]))
		return freq_dict