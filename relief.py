import utils
import numpy as np
from tqdm import tqdm


class Relief():
    """
    Relief algorithm class - use for estimating the quality of attributes.
        Parameters:
            m: arbritary variable
            aglorithm: {Relief, ReliefF, RReliefF} 

    """

    def __init__(self, m: int, algorithm: str, text_distance_funct: str = "fast", k=10):
        # regarding the algorithm
        self.m = m
        self.text_distance_funct = text_distance_funct
        self.algorithm = algorithm
        self.k = k
        # regarding the data
        self.num_samples = None
        self.num_attributes = None
        self.data = None
        self.target = None
        self.num_classes = None
        self.data_info = {'column_types': [], 'max': [], 'min': []}
        self.text_embeddings = {}
        # regarding the output
        self.weights = None
        # for Reliefs
        self.instances = None
        self.priors = None
        self.model = None
        # regarding plots
        self.history = None

    def sanity_check(self):
        """
        Make sure that Relief algorithm has the valid inputs to succesfully execute.
        """
        # regarding missing data
        missing_values = self.data.isna().sum().sum()
        n_classes = self.target.nunique()
        if missing_values:
            print("The dataset contains missing values, total number: ",
                  missing_values)
            return False
        # regarding the number of classes
        elif n_classes != 2:
            print(
                "The task is not a classification problem, exact number of classes: ", n_classes)
            return False
        else:
            print("Clean dataset")
            return True

    def diff(self, attribute, sample1, sample2):
        if self.data_info['column_types'][attribute] == "numerical":
            return self.diff_numerical(attribute, sample1, sample2)
        elif self.data_info['column_types'][attribute] == "nominal":
            return self.diff_nominal(attribute, sample1, sample2)
        elif self.data_info['column_types'][attribute] == "text":
            return self.diff_text(attribute, sample1, sample2)
        else:
            print("data type not implemented")
            return -10000

    def diff_numerical(self, attribute, sample1, sample2):
        numerator = np.abs(
            self.data.iloc[sample1][attribute] - self.data.iloc[sample2][attribute])
        denominator = self.data_info['max'][attribute] - \
            self.data_info['min'][attribute]
        return numerator/denominator

    def diff_nominal(self, attribute, sample1, sample2):
        if self.data.iloc[sample1][attribute] == self.data.iloc[sample2][attribute]:
            return 0
        else:
            return 1

    def diff_text(self, attribute, sample1, sample2):
        return utils.cosine_distance(self.text_embeddings[attribute][sample1], self.text_embeddings[attribute][sample2])

    # if one instance is missing e.g.(I1) has unknown value:
    # # diff (A, I1, I2) = 1 - P(value(A, I2)|class(I1))
    # if both instances have unknown value
    # diff(A,I1,I2) = 1− #values(A)∑V  P(V|class(I1))×P(V|class(I2))

    # conditional probabilities are approximated with relative frequencies from the training set.
    # given a dataframe find the relative frequency of that particular value for that particular attribute

    def diff_reliefF(self, attribute, sample1, sample2):
        # check if the attribute is string. since they must be completed. it doesn't matter which sample to select.
        if isinstance(self.data.iloc[sample1][attribute], str):
            return self.diff(attribute, sample1, sample2)
        # both missing
        if np.isnan(self.data.iloc[sample1][attribute]) and np.isnan(self.data.iloc[sample2][attribute]):
            return self.diff_2_missing(attribute, sample1, sample2)
        elif np.isnan(self.data.iloc[sample1][attribute]):
            return self.diff_1_missing(attribute, sample1, sample2)
        elif np.isnan(self.data.iloc[sample2][attribute]):
            return self.diff_1_missing(attribute, sample2, sample2)
        else:
            return self.diff(attribute, sample1, sample2)

    def relative_frequency(self, missing_sample_idx, attribute, value):
        # indexes of the sample with sample class label as the missing sample
        indexes = self.instances[self.target.iloc[missing_sample_idx]]
        # subset of the dataframe with same class label
        df = self.data.iloc[indexes]
        denom = len(df)
        rf = df.iloc[:, attribute].value_counts()[value] / denom
        return rf

    def diff_1_missing(self, attribute, sample_missing, sample_present):
        """
        sample_missing: the index of the sample that the value is missing
        sample_present: the index of the sample that the value is present
        """
        # use the value of the known sample
        value = self.data.iloc[sample_present][attribute]
        diff = 1 - self.relative_frequency(sample_missing, attribute, value)
        return diff

    def diff_2_missing(self, attribute, sample1, sample2):
        """
            sample1: index of sample1 in the orginal dataframe
            sample2: index of sample2 in the orginal dataframe
            attribute: column of the original dataframe
        """
        set_of_values = set(self.data.iloc[:, attribute])

        diff = sum([self.relative_frequency(sample1, attribute, v) *
                    self.relative_frequency(sample2, attribute, v) for v in set_of_values])
        return 1 - diff

    def distance_between_samples(self, sample1, sample2, algorithm="relief"):
        if algorithm == "relief":
            distance = 0
            for a in range(self.num_attributes):
                distance += self.diff(a, sample1, sample2)
            return distance
        else:
            distance = 0
            for a in range(self.num_attributes):
                distance += self.diff_reliefF(a, sample1, sample2)
            return distance

    def create_embeddings(self):
        for i, att_type in enumerate(self.data_info['column_types']):
            if att_type == "text":
                # select the attribute i where the type is text
                sentences = list(self.data.iloc[:, i])
                sentence_embeddings = self.model.encode(sentences)
                # store the sentences embeddings for the original text
                self.text_embeddings[i] = sentence_embeddings
                print(
                    f"The text embeddings for column {self.data.columns[i]} were created.")

    def get_nearest(self, sample_idx, label):
        """
        Given a sample index and the label
        Returns:
            the index of the smallest distances in the array of the samples.
        """
        nearest_array = []
        for idx in self.instances[label]:
            if sample_idx == idx:
                distance = np.inf
            else:
                distance = self.distance_between_samples(sample_idx, idx)
            nearest_array.append((idx, distance))
        return utils.get_min_row_first_element(nearest_array)

    def get_k_nearest(self, sample_idx, label):
        nearest_array = []
        for idx in self.instances[label]:
            if sample_idx == idx:
                distance = np.inf
            else:
                distance = self.distance_between_samples(
                    sample_idx, idx, algorithm="reliefF")
            nearest_array.append((idx, distance))
        return utils.get_k_smallest_first_elements(nearest_array, self.k)

    def update_misses(self, a, label, random_sample, misses):
        """
        function that returns the value for updating the attributes weights parametrizes by the k nearest miss.
        """
        result = 0
        for miss_class in self.priors.keys():
            if miss_class == label:
                continue
            else:
                inner_multi = self.priors[miss_class]/(1 - self.priors[label])
                inner_sum = sum([self.diff_reliefF(
                    a, random_sample, miss) for miss in misses[miss_class]])
                result += inner_multi*inner_sum/(self.m*self.k)
        return result

    def _get_KNN_RreliefF(self, Ridx):
        """
        Function that return the 
        Parameters:
            self.k = number of Nearest Neighbors
            Ridx = index sample to get the nearest from
        Returns:
            nn = a list of index of the Nearest Neighbors.
        """
        distance_to_target = [(idj, self.distance_between_samples(
            Ridx, idj)) for idj in range(self.num_samples) if idj != Ridx]
        nn = utils.get_k_smallest_first_elements(distance_to_target, self.k)
        return nn

    def get_k_nearest_RreliefF(self, idx):
        """
        Function that return the indexes of the k nearest neighors of the idx sample
        Parameters:
            idx: index of the sample
        Returns:
            nn = array of index of the Nearest Neighbors
        """
        # for Regresion we use the target values to obtain the nn.
        diff_in_values = (self.target - self.target[idx]).abs()
        # remove target index
        diff_in_values.drop(labels=idx, inplace=True)
        # select the index of the k smallest
        nn = diff_in_values.nsmallest(self.k).index
        return nn

    def _initialize_weights(self):
        return np.zeros(self.num_attributes)

    def _separate_dataset(self):
        """
        given the labels separate the dataset into each label category.
        the return array will be a 2d array of indices that will reference 
        the indices of the sample in the self.data array
        """
        instances = [[] for i in range(self.num_classes)]
        for idx in range(self.num_samples):
            label = self.target.iloc[idx]
            instances[label].append(idx)
        return instances

    def _Relief(self):
        """
        Estimates the quality of the attributes given a binary classification problem 
        the data must be complete since it doesn't work with missing data. 
        """
        self.num_classes = 2
        self.instances = self._separate_dataset()
        self.weights = self._initialize_weights()

        # generate random indices for the algorithm run, this will serve as indices to choose the random sample m times.
        indices = np.random.randint(self.num_samples, size=self.m)
        # loop over the m indices
        for count, idx in tqdm(enumerate(indices)):
            random_idx, label = idx, self.target.iloc[idx]
            # find nearest hit H and nearest miss M
            nearest_hit = self.get_nearest(idx, label)
            # if hit=0 then miss=1 else hit=1 miss=0
            nearest_miss = self.get_nearest(idx, 1 - label)

            for a in range(self.num_attributes):
                self.weights[a] = self.weights[a] - self.diff(
                    a, random_idx, nearest_hit)/self.m + self.diff(a, random_idx, nearest_miss)/self.m
            self.history[count] = self.weights

    def _ReliefF(self):
        """
        Extension of Relief algorithm, not limited to binary classification
        more robust and can deal with incomplete and noisy data. 
        search K nearest neighbors from the same class Hits and Misses to estimate the quality.

        selection of k hits an misses is the basic difference to Relief.
        user defined parameter K control the locality of the estimates.
        For most puposes it can safely set K =10 Kononenko, 1994

        It works with incomplete data, it handles missing data from a probabilistic approach. 
        """
        self.num_classes = self.target.nunique()
        self.instances = self._separate_dataset()
        self.weights = self._initialize_weights()

        # get the prior probabilities of classes.
        self.priors = utils.get_class_probabilities(self.target)
        # generate random indices for the algorithm run, this will serve as indices to choose the random sample m times.
        indices = np.random.randint(self.num_samples, size=self.m)
        # loop over the m indices
        for count, idx in tqdm(enumerate(indices)):
            #print("indices of the random sample selected: ", idx)
            random_idx, label = idx, self.target.iloc[idx]
            # find K nearest hits and neares misses
            k_nearest_hits = self.get_k_nearest(random_idx, label)
            #print("indices of the k hits: ", k_nearest_hits)
            # find K neatest miss for each class C different than label.
            misses = {}
            for miss_class in range(self.num_classes):
                if miss_class == label:
                    continue
                else:
                    misses[miss_class] = self.get_k_nearest(
                        random_idx, miss_class)

            #print("indices of the k misses: ", misses)

            for a in range(self.num_attributes):
                self.weights[a] = self.weights[a] - sum([self.diff_reliefF(a, random_idx, hit) for hit in k_nearest_hits])/(self.m*self.k) \
                    + self.update_misses(a, label, random_idx, misses)
            self.history[count] = self.weights

            #print(f"the history of the weights for iteration {count} is : {self.history[count]}")

    def _RReliefF(self):
        """
        Relief actually calculates an approximation of the following differences in probabilities.
        W[A] = P(diff. value of A|nearest inst. from diff. class) − P(diff. value of A|nearest inst. from same class)

        Algorithm use for regression models where the target value is continous. thus Nearest H&M can not be used. 
        To solve this problem a kind of probablity that the predicted values of 2 instances are different is introduced.
        It can be modelled with the relative distance between the predicted (class) values of 2 instances. 

        N_dA = P(different attributes | nearest instances)
        N_dC = P(different prediction | nearest instances)
        N_dC_dA = P(diff.prediction | diff. attributes & nearest instances)

        We use Bayes' rule to update the estimate of the weights

        W[A] = N_dC_dA*N_dA/N_dC  - (1-N_dC_dA)*N_dA/(1-N_dC)

        To stick with a probabilistic interpretation of the results we normalize the contribution of each K nearest instances by dividing
        it with the sum of all K contributions. 
        We can use ranks or actual distances, but using actual distances makes it problem dependent. 
        While using ranks we assure that the nearest instance always has the same impact on the weights.
        But if using ranks we might lose the intrinsic self normalization contained in the distances between instances of the given problem
        """
        # generate random indices for the algorithm run, this will serve as indices to choose the random sample m times.
        self.weights = self._initialize_weights()
        N_dC = 0
        N_dA = np.zeros(self.num_attributes)
        N_dC_dA = np.zeros(self.num_attributes)

        indices = np.random.randint(self.num_samples, size=self.m)
        for count, idx in tqdm(enumerate(indices)):
            # indexes of nn
            nearest_instances = self.get_k_nearest_RreliefF(idx)
            d1_sum = sum([1/self.distance_between_samples(idx, Il)
                          for Il in nearest_instances])
            for Ij in nearest_instances:
                d1_i_j = 1/self.distance_between_samples(idx, Ij)
                dij = d1_i_j/d1_sum
                N_dC += np.abs(self.target.iloc[idx] -
                               self.target.iloc[Ij])*dij

                for a in range(self.num_attributes):
                    N_dA[a] += self.diff(a, idx, Ij)*dij
                    N_dC_dA[a] += np.abs(self.target.iloc[idx] -
                                         self.target.iloc[Ij])*self.diff(a, idx, Ij)*dij

            for a in range(self.num_attributes):
                self.weights[a] = N_dC_dA[a]/N_dC - \
                    (N_dA[a] - N_dC_dA[a])/(count - N_dC)
            self.history[count] = self.weights

    def fit(self, X, y, column_types):
        self.data_info['column_types'] = column_types
        self.data_info['max'] = X.max()
        self.data_info['min'] = X.min()
        self.data = X
        self.target = y

        self.num_samples = self.data.shape[0]
        self.num_attributes = self.data.shape[1]
        self.history = np.zeros((self.m, self.num_attributes))
        if "text" in self.data_info['column_types']:
            self.model = utils.initialize_model(self.text_distance_funct)
            # encode each sentences and replace the column with the embeddings value.
            self.create_embeddings()

        if self.algorithm == "Relief":
            if self.sanity_check():
                self._Relief()

        elif self.algorithm == "ReliefF":
            print("Starting ReliefF...")
            self._ReliefF()
            print("ReliefF Completed...")

        elif self.algorithm == "RReliefF":
            self._RReliefF()
