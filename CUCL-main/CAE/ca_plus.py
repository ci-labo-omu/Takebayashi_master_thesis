#
# Copyright (c) 2023 Naoki Masuyama (masuyama@omu.ac.jp)
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#

from collections.abc import Iterable, Iterator
from itertools import chain, count, repeat, compress

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.estimator_checks import check_estimator
from typing import Iterable, Iterator

class CAplus(BaseEstimator):
    """ CIM-based Adaptive Resonance Theory plus (CA+)"""

    def __init__(
            self,
            G_=nx.Graph(),
            dim_=None,
            num_signal_=0,
            V_thres_=1.0,
            sigma_=None,
            n_clusters_=0,
            active_node_idx_=None,
            flag_set_lambda_=False,
            n_init_data_=256,
            n_active_nodes_=np.inf,
            div_mat_=None,
            div_threshold_=1.0e-6,
            # div_threshold_=0.01,
            div_lambda_=np.inf,

            max_nodes_=np.inf  # for buffer
    ):

        self.G_ = G_  # network
        self.dim_ = dim_  # Number of variables in an instance
        self.num_signal_ = num_signal_  # Counter for training instances
        self.sigma_ = sigma_  # An estimated sigma for CIM
        self.V_thres_ = V_thres_  # Similarity threshold
        self.n_clusters_ = n_clusters_  # Number of clusters
        self.active_node_idx_ = active_node_idx_  # Indexes of active nodes
        self.flag_set_lambda_ = flag_set_lambda_  # Flag for setting \lambda
        self.n_init_data_ = n_init_data_  # Number of signals for initialization of sigma
        self.n_active_nodes_ = n_active_nodes_  # Number of buffer nodes for calculating \sigma
        self.div_mat_ = div_mat_  # A matrix for diversity via determinants
        self.div_threshold_ = div_threshold_  # A threshold for diversity via determinants
        self.div_lambda_ = div_lambda_  # \lambda determined by diversity via determinants

        self.max_nodes_ = max_nodes_  # The number of maximum nodes = buffer size

    def fit(self, x: np.ndarray):
        """
        train data in batch manner
        :param x: array-like or ndarray
        """
        # avoiding IndexError caused by a complex-valued label
        # Comment-in when check_estimator() is used
        # y = y.real.astype(int)

        self.initialization(x)

        for signal in x:
            self.input_signal(signal, x)  # training a network

        return self

    def predict(self, x: np.ndarray):
        """
        predict cluster index for each sample.
        :param x: array-like or ndarray
        :rtype list:
            cluster index for each sample.
        """

        self.labels_ = self.__labeling_sample_for_clustering(x)

        return self.labels_

    def fit_predict(self, x: np.ndarray):
        """
        train data and predict cluster index for each sample.
        :param x: array-like or ndarray
        :rtype list:
            cluster index for each sample.
        """

        return self.fit(x).__labeling_sample_for_clustering(x)

    def initialization(self, x: np.ndarray):
        """
        Initialize parameters
        :param x: array-like or ndarray
        """
        # set graph
        if len(list(self.G_.nodes)) == 0:
            self.G_ = nx.Graph()

        # set dimension of x
        if self.dim_ is None:
            self.dim_ = x.shape[1]

    def input_signal(self, signal: np.ndarray, x: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the state. So the function does batch training.
        :param signal: A new input signal
        :param x: array-like or ndarray
            data
        """

        if self.num_signal_ == x.shape[0]:
            self.num_signal_ = 1
        else:
            self.num_signal_ += 1

        # # delete nodes based on winning_counts
        if self.G_.number_of_nodes() == self.max_nodes_:
            deleted_node_list = self.__delete_nodes(deletion_percentage=0.0)
            # deleted_node_list = self.__delete_nodes_by_winning_counts()
            self.__delete_active_node_index(deleted_node_list)

        if self.num_signal_ == 1 and self.G_.number_of_nodes() == 0:
            self.__calculate_sigma_by_active_nodes(x[0:self.n_init_data_, :], None)  # set init \sigma

        if (self.flag_set_lambda_ is False or self.G_.number_of_nodes() < self.n_active_nodes_) and self.G_.number_of_nodes() < self.max_nodes_:
            new_node_idx = self.__add_node(signal)
            self.__update_active_node_index(signal, new_node_idx)

            # setup initial n_active_nodes_, div_lambda_, and V_thres_
            self.__setup_init_params()

        else:
            node_list, cim = self.__calculate_cim(signal)
            s1_idx, s1_cim, s2_idx, s2_cim = self.__find_nearest_node(node_list, cim)
            # print("s1_index", s1_idx)

            if (self.V_thres_ < s1_cim or self.G_.number_of_nodes() < self.n_active_nodes_) and self.G_.number_of_nodes() < self.max_nodes_:
                new_node_idx = self.__add_node(signal)
                self.__update_active_node_index(signal, new_node_idx)
                self.__calculate_sigma_by_active_nodes(None, new_node_idx)
            else:
                self.__update_s1_node(s1_idx, signal)
                self.__update_active_node_index(signal, s1_idx)

                if self.V_thres_ >= s2_cim:
                    self.__update_s2_node(s2_idx, signal)

        # delete nodes based on winning_counts
        # if self.G_.number_of_nodes() >= self.max_nodes_ + 1:
        #     deleted_node_list = self.__delete_nodes(deletion_percentage=0.2)
        #     # deleted_node_list = self.__delete_nodes_by_winning_counts()
        #     self.__delete_active_node_index(deleted_node_list)


    def __setup_init_params(self):
        """
        Setup initial n_active_nodes_, div_lambda_, and V_thres_
        """

        if self.G_.number_of_nodes() >= 2 and self.flag_set_lambda_ is False:
            # calculate n_active_nodes_ and div_lambda_ based on diversity via determinants
            self.__setup_n_active_nodes_and_div_lambda()

        if self.G_.number_of_nodes() == self.n_active_nodes_:
            self.flag_set_lambda_ = True

            # estimate \sigma by using active nodes
            self.__calculate_sigma_by_active_nodes()

            # overwrite \sigma of all nodes
            [nx.set_node_attributes(self.G_, {k: {'sigma': self.sigma_}}) for k in list(self.G_.nodes)]

            # get similarity threshold
            self.__calculate_threshold_by_active_nodes()

    def __setup_n_active_nodes_and_div_lambda(self):
        """
        Setup n_active_nodes_ and div_lambda_ by Diversity of nodes.
        https://proceedings.neurips.cc/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> CAplusnet.n_active_nodes_ = np.inf
        >>> CAplusnet.div_lambda_ = np.inf
        >>> CAplusnet.div_threshold_ = 1.0e-6
        >>> CAplusnet._CAplus__setup_n_active_nodes_and_div_lambda()

        First, a pairwise CIM-based similarity matrix is calculated.
        >>> CAplusnet.div_mat_
        array([[1.        , 0.86058076],
               [0.86058076, 1.        ]])

        Then, determinant of div_mat_ is calculated.
        >>> div_cim = np.linalg.det(np.exp(CAplusnet.div_mat_))
        >>> div_cim
        1.7980373454447525

        In this case, div_cim < self.div_threshold_ is not satisfied.
        Thus, n_active_nodes_ and div_lambda_ are not updated.
        >>> CAplusnet.n_active_nodes_
        inf
        >>> CAplusnet.div_lambda_
        inf

        Adding a new node until div_cim < self.div_threshold_ is satisfied.
        >>> CAplusnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3
        >>> CAplusnet._CAplus__setup_n_active_nodes_and_div_lambda()
        >>> CAplusnet.div_mat_
        array([[1.        , 0.86058076, 0.79504658],
               [0.86058076, 1.        , 0.97550498],
               [0.79504658, 0.97550498, 1.        ]])
        >>> div_cim = np.linalg.det(np.exp(CAplusnet.div_mat_))
        >>> div_cim
        0.21027569040368288

        Adding a new node until div_cim < self.div_threshold_ is satisfied.
        >>> CAplusnet.G_.add_node(3, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node4
        >>> CAplusnet._CAplus__setup_n_active_nodes_and_div_lambda()
        >>> div_cim = np.linalg.det(np.exp(CAplusnet.div_mat_))
        >>> div_cim
        -1.3286163759107123e-32
        """

        nodes_list = list(self.G_.nodes)
        _, correntropy = self.__calculate_correntropy(self.G_.nodes[nodes_list[-1]]['weight'])

        if self.G_.number_of_nodes() == 2:
            self.div_mat_ = np.array([[correntropy[1], correntropy[0]], [correntropy[0], correntropy[1]]])
        else:
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy[0:self.div_mat_.shape[1]],
                                      axis=0)
            self.div_mat_ = np.insert(self.div_mat_, self.div_mat_.shape[1], correntropy, axis=1)

        div_cim = np.linalg.det(self.div_mat_)
        # div_cim = np.linalg.det(np.exp(self.div_mat_))

        # if div_cim < self.div_threshold_ and self.G_.number_of_nodes() >= self.n_init_data_:
        if div_cim < self.div_threshold_:
            self.n_active_nodes_ = self.G_.number_of_nodes()
            self.div_lambda_ = self.n_active_nodes_ * 2

    def __calculate_sigma_by_active_nodes(self, weight: np.ndarray = None, new_node_idx: int = None):
        """
        Calculate \sigma for CIM basd on active nodes

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.dim_ = 2
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> CAplusnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3

        Depending on active nodes, a value of sigma will be changed.
        >>> CAplusnet.active_node_idx_ = [0, 1]
        >>> CAplusnet._CAplus__calculate_sigma_by_active_nodes()
        >>> CAplusnet.sigma_
        0.2834822362263465
        >>> CAplusnet.active_node_idx_ = [0, 1, 2]
        >>> CAplusnet._CAplus__calculate_sigma_by_active_nodes()
        >>> CAplusnet.sigma_
        0.2920448418024727

        A sigma for a new node can be set by using the current sigma.
        >>> CAplusnet.G_.add_node(3, weight=[0.0, 0.1], winning_counts=1, sigma=1.0)  # node4
        >>> new_node_idx = 3
        >>> CAplusnet._CAplus__calculate_sigma_by_active_nodes(None, new_node_idx)
        >>> nx.get_node_attributes(CAplusnet.G_, 'sigma')
        {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.2920448418024727}
        """

        if weight is None:
            active_node_idx_ = list(self.active_node_idx_)
            n_selected_weights = np.minimum(len(active_node_idx_), self.n_active_nodes_)
            selected_weights = list(
                self.__get_node_attributes_from('weight', active_node_idx_[0:int(n_selected_weights)]))
            std_weights = np.std(selected_weights, axis=0, ddof=1)
        else:
            selected_weights = weight
            std_weights = np.std(weight, axis=0, ddof=1)
        np.putmask(std_weights, std_weights == 0.0, 1.0e-6)  # If value=0, add a small value for avoiding an error.

        # Silverman's Rule
        a = np.power(4 / (2 + self.dim_), 1 / (4 + self.dim_))
        b = np.power(np.array(selected_weights).shape[0], -1 / (4 + self.dim_))
        s = a * std_weights * b
        self.sigma_ = np.median(s)

        if new_node_idx is not None:
            nx.set_node_attributes(self.G_, {new_node_idx: {'sigma': self.sigma_}})

    def __calculate_cim(self, signal: np.ndarray):
        """
        Calculate CIM between a signal and nodes.
        Return indexes of nodes and cim value between a signal and nodes

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = np.array([0, 0])

        Return an index and a value of the cim between a node and a signal.
        >>> CAplusnet._CAplus__calculate_cim(signal)
        ([0], array([0.2474779]))

        If there are multiple nodes, return multiple indexes and values of the cim.
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> CAplusnet._CAplus__calculate_cim(np.array([0, 0]))
        ([0, 1], array([0.2474779 , 0.49887522]))
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.sqrt(1 - np.mean(c, axis=1))

    def __calculate_correntropy(self, signal: np.ndarray):
        """
        Calculate CIM between a signal and nodes.
        Return indexes of nodes and cim value between a signal and nodes

        Setup
        >>> net = CAplus()
        >>> net.G_ = nx.Graph()
        >>> net.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = np.array([0, 0])

        Return an index and a value of the cim between a node and a signal.
        >>> net._CAplus__calculate_cim(signal)
        ([0], array([0.2474779]))

        If there are multiple nodes, return multiple indexes and values of the cim.
        >>> net.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> net._CAplus__calculate_cim(np.array([0, 0]))
        ([0, 1], array([0.2474779 , 0.49887522]))
        """
        node_list = list(self.G_.nodes)
        weights = list(self.__get_node_attributes_from('weight', node_list))
        sigma = list(self.__get_node_attributes_from('sigma', node_list))
        c = np.exp(-(signal - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigma)) ** 2))
        return node_list, np.mean(c, axis=1)

    def __add_node(self, signal: np.ndarray) -> int:
        """
        Add a new node to G with winning count, sigma, and label_counts.
        Return an index of the new node.

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.sigma_ = 0.5
        >>> CAplusnet.init_label_list_ = np.array([0,0])

        Add the 1st node to G with label=0
        >>> signal = np.array([1,2])
        >>> new_node_idx = CAplusnet._CAplus__add_node(signal)
        >>> new_node_idx
        0
        >>> list(CAplusnet.G_.nodes.data())
        [(0, {'weight': array([1, 2]), 'winning_counts': 1, 'sigma': 0.5})]

        Add the 2nd node to G with label=1
        >>> signal = np.array([3,4])
        >>> new_node_idx = CAplusnet._CAplus__add_node(signal)
        >>> new_node_idx
        1
        >>> list(CAplusnet.G_.nodes.data())
        [(0, {'weight': array([1, 2]), 'winning_counts': 1, 'sigma': 0.5}), (1, {'weight': array([3, 4]), 'winning_counts': 1, 'sigma': 0.5})]
        """
        if len(self.G_.nodes) == 0:  # for the first node
            new_node_idx = 0
        else:
            new_node_idx = max(self.G_.nodes) + 1

        # Generate node
        self.G_.add_node(new_node_idx, weight=signal, winning_counts=1, sigma=self.sigma_)

        return new_node_idx

    def __update_active_node_index(self, signal, winner_idx):
        if self.active_node_idx_ is None:
            self.active_node_idx_ = np.array([winner_idx])
        else:
            delete_idx = np.where(self.active_node_idx_ == winner_idx)
            self.active_node_idx_ = np.delete(self.active_node_idx_, delete_idx)
            self.active_node_idx_ = np.append(winner_idx, self.active_node_idx_)

    def __delete_active_node_index(self, deleted_node_list: list):
        delete_idx = [np.where(self.active_node_idx_ == deleted_node_list[k]) for k in range(len(deleted_node_list))]
        self.active_node_idx_ = np.delete(self.active_node_idx_, delete_idx)

    def __calculate_threshold_by_active_nodes(self) -> float:
        """
        Calculate a similarity threshold by using active nodes.
        Return a similarity threshold

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node1
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)  # node2
        >>> CAplusnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)  # node3
        >>> CAplusnet.active_node_idx_ = [0, 1, 2]
        >>> CAplusnet.n_active_nodes_ = 10

        Return a mean of the minimum pairwise cims among nodes 1, 2, and 3.
        >>> CAplusnet._CAplus__calculate_threshold_by_active_nodes()
        >>> CAplusnet.V_thres_
        0.22880218578964573

        A simple explanation of this function is as follows:
        First, we calculate cim between nodes 1-2, and 1-3, and take min of cims.
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> signal = np.array([0.1, 0.5])
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> CAplusnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)
        >>> _, cims = CAplusnet._CAplus__calculate_cim(signal)
        >>> np.min(cims)
        0.37338886146591654

        Second, we calculate cim between nodes 2-1, and 2-3, and take min of cims.
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> signal = [0.9, 0.6]
        >>> CAplusnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=1, sigma=1.0)
        >>> _, cims = CAplusnet._CAplus__calculate_cim(signal)
        >>> np.min(cims)
        0.1565088479515103

        Third, we calculate cim between nodes 3-1, and 3-2, and take min of cims.
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> signal = [1.0, 0.9]
        >>> _, cims = CAplusnet._CAplus__calculate_cim(signal)
        >>> np.min(cims)
        0.1565088479515103

        A mean of them is the same value as return from the function.
        >>> np.mean([0.37338886146591654, 0.1565088479515103, 0.1565088479515103])
        0.22880218578964573
        """

        active_node_idx_ = list(self.active_node_idx_)
        n_selected_weights = np.minimum(len(active_node_idx_), self.n_active_nodes_)
        selected_weights = list(self.__get_node_attributes_from('weight', active_node_idx_[0:int(n_selected_weights)]))
        cims = [self.__calculate_cim(w)[1] for w in selected_weights]  # Calculate a pairwise cim among nodes
        [np.putmask(cims[k], cims[k] == 0.0, 1.0) for k in range(len(cims))]  # Set cims[k][k] = 1.0
        self.V_thres_ = np.mean([np.min(cims[k]) for k in range(len(cims))])

    def __find_nearest_node(self, node_list: list, cim: np.ndarray):
        """
        Get 1st and 2nd nearest nodes from a signal.
        Return indexes and weights of the 1st and 2nd nearest nodes from a signal.

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)

        If there is only one node, return an index and the cim value of the 1st nearest node.
        In this case, for the 2nd nearest node, an index is the same as the 1st nearest node and its value is inf.
        >>> node_list = [0]
        >>> cim = np.array([0.5])
        >>> CAplusnet._CAplus__find_nearest_node(node_list, cim)
        (0, 0.5, 0, inf)

        If there are two nodes, return an index and the cim value of the 1st and 2nd nearest nodes.
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=1, sigma=1.0)
        >>> node_list = [0, 1]
        >>> cim = np.array([0.5, 0.9])
        >>> CAplusnet._CAplus__find_nearest_node(node_list, cim)
        (0, 0.5, 1, 0.9)
        """

        if len(node_list) == 1:
            node_list = node_list + node_list
            cim = np.array(list(cim) + [np.inf])

        idx = np.argsort(cim)
        return node_list[idx[0]], cim[idx[0]], node_list[idx[1]], cim[idx[1]]

    def __update_s1_node(self, idx, signal):
        """
        Update weight for s1 node

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.sigma_ = 1.0
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=CAplusnet.sigma_)
        >>> signal = np.array([0,0])
        >>> s1_idx = 0
        >>> CAplusnet.G_.nodes[s1_idx]
        {'weight': [0.1, 0.5], 'winning_counts': 1, 'sigma': 1.0}

        Update weight, winning_counts, and label_counts of s1 node.
        >>> CAplusnet._CAplus__update_s1_node(s1_idx, signal)
        >>> CAplusnet.G_.nodes[s1_idx]
        {'weight': array([0.05, 0.25]), 'winning_counts': 2, 'sigma': 1.0}
        """
        # update weight and winning_counts
        weight = self.G_.nodes[idx].get('weight')
        new_winning_count = self.G_.nodes[idx].get('winning_counts') + 1
        new_weight = weight + (signal - weight) / new_winning_count
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight, 'winning_counts': new_winning_count}})

    def __get_node_attributes_from(self, attr: str, node_list: Iterable[int]) -> Iterator:
        """
        Get an attribute of nodes in G

        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=1.0)  # node 0
        >>> CAplusnet.G_.add_node(1, weight=[0.9, 0.6], winning_counts=2, sigma=2.0)  # node 1
        >>> CAplusnet.G_.add_node(2, weight=[1.0, 0.9], winning_counts=3, sigma=3.0)  # node 2
        >>> node_list = list(CAplusnet.G_.nodes)
        >>> node_list
        [0, 1, 2]

        Get weight of node.
        >>> list(CAplusnet._CAplus__get_node_attributes_from('weight', node_list))
        [[0.1, 0.5], [0.9, 0.6], [1.0, 0.9]]

        Get winning_counts of node.
        >>> list(CAplusnet._CAplus__get_node_attributes_from('winning_counts', node_list))
        [1, 2, 3]

        Get sigma of node.
        >>> list(CAplusnet._CAplus__get_node_attributes_from('sigma', node_list))
        [1.0, 2.0, 3.0]
        """
        att_dict = nx.get_node_attributes(self.G_, attr)
        return map(att_dict.get, node_list)

    def __update_s2_node(self, idx, signal):
        """Update weight for s2 node
        Setup
        >>> CAplusnet = CAplus()
        >>> CAplusnet.G_ = nx.Graph()
        >>> CAplusnet.sigma_ = 1.0
        >>> CAplusnet.G_.add_node(0, weight=[0.1, 0.5], winning_counts=1, sigma=CAplusnet.sigma_)
        >>> signal = np.array([0,0])
        >>> s2_idx = 0
        >>> CAplusnet.G_.nodes[s2_idx]
        {'weight': [0.1, 0.5], 'winning_counts': 1, 'sigma': 1.0}

        Update weight of s2 node
        Because a learning coefficient is different from __update_s1_node(), a value of weight is different.
        In addition, winning_counts of s2 node is not updated.
        >>> CAplusnet._CAplus__update_s2_node(s2_idx, signal)
        >>> CAplusnet.G_.nodes[s2_idx]
        {'weight': array([0.099, 0.495]), 'winning_counts': 1, 'sigma': 1.0}
        """
        weight = self.G_.nodes[idx].get('weight')
        winning_counts = self.G_.nodes[idx].get('winning_counts')
        new_weight = weight + (signal - weight) / (100 * winning_counts)
        nx.set_node_attributes(self.G_, {idx: {'weight': new_weight}})

    def __delete_nodes(self, deletion_percentage=0.1) -> list:
        """
        Delete nodes based on winning_counts.
        Returns a list of nodes to be deleted.
        :param deletion_percentage: The percentage of nodes to delete, default is 10%.
        """
        # Get winning_counts for all nodes
        node_winning_counts = [(node, data['winning_counts']) for node, data in self.G_.nodes(data=True)]

        # Sort nodes based on winning_counts (ascending)
        node_winning_counts.sort(key=lambda x: x[1])

        # Calculate the number of nodes to remove (based on the specified percentage)
        num_nodes_to_remove = int(len(node_winning_counts) * deletion_percentage)

        # Select nodes to be removed
        to_be_removed = [node for node, _ in node_winning_counts[:num_nodes_to_remove]]

        # Remove selected nodes
        self.G_.remove_nodes_from(to_be_removed)

        return to_be_removed

    # def __delete_nodes(self, deletion_percentage=0.1) -> list:
    #     """
    #     Delete nodes based on winning_counts if they are 1 or less.
    #     Returns a list of nodes to be deleted.
    #     :param deletion_percentage: The percentage of nodes to delete, default is 10%.
    #     """
    #     # Get winning_counts for all nodes
    #     node_winning_counts = [(node, data['winning_counts']) for node, data in self.G_.nodes(data=True)]
    #
    #     # Filter nodes where winning_counts are 1 or less
    #     filtered_nodes = [(node, count) for node, count in node_winning_counts if count <= 1]
    #
    #     # Sort filtered nodes based on winning_counts (ascending)
    #     filtered_nodes.sort(key=lambda x: x[1])
    #
    #     # Calculate the number of nodes to remove (based on the specified percentage)
    #     num_nodes_to_remove = int(len(filtered_nodes) * deletion_percentage)
    #
    #     # Select nodes to be removed
    #     to_be_removed = [node for node, _ in filtered_nodes[:num_nodes_to_remove]]
    #
    #     # Remove selected nodes
    #     self.G_.remove_nodes_from(to_be_removed)
    #
    #     return to_be_removed

    def __delete_nodes_by_winning_counts(self) -> list:
        """
        Delete nodes where winning_counts are 1 or less.
        Returns a list of nodes to be deleted.
        """
        # Identify nodes with winning_counts of 1 or less
        to_be_removed = [node for node, data in self.G_.nodes(data=True) if data['winning_counts'] <= 1]

        # Remove identified nodes
        self.G_.remove_nodes_from(to_be_removed)

        return to_be_removed

    def __labeling_sample_for_clustering(self, x: np.ndarray) -> np.ndarray:
        """
        A label of testing sample is determined by connectivity of nodes.
        Labeled samples should be evaluated by using clustering metrics.

        """
        # get cluster of nodes and order of nodes
        # compute cim between x and nodes
        weights = list(self.__get_node_attributes_from('weight', list(self.G_.nodes)))
        sigmas = list(self.__get_node_attributes_from('sigma', list(self.G_.nodes)))
        c = [np.exp(-(x[k, :] - np.array(weights)) ** 2 / (2 * np.mean(np.array(sigmas)) ** 2)) for k in range(len(x))]
        cim = [np.sqrt(1 - np.mean(c[k], axis=1)) for k in range(len(x))]

        # get indexes of the nearest neighbor
        nearest_node_idx = np.argmin(cim, axis=1)

        return nearest_node_idx

    def plotting_ca_plus(self, x: np.ndarray = None, fig_name=None):
        fig, ax = plt.subplots()
        fig.tight_layout()
        if fig_name is not None:
            plt.title(fig_name)
        if x is not None:
            plt.plot(x[:, 0], x[:, 1], 'cx', zorder=1)
        nx.draw(self.G_, pos=nx.get_node_attributes(self.G_, 'weight'), node_size=40, node_color='r', with_labels=False,
                ax=ax)
        ax.set_axis_on()
        ax.set_axisbelow(True)
        ax.set_aspect('equal')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.grid(True)
        plt.show()


class ClusterCAplus(CAplus, ClusterMixin):
    pass


if __name__ == '__main__':
    # https://docs.python.org/3.10/library/doctest.html
    import doctest

    doctest.testmod()

    # check_estimator(CAplus())
