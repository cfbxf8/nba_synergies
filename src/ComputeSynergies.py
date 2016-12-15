import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
import random
import time
import cPickle as pickle
import math
from datetime import datetime, timedelta
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, timeit, connect_sql, subset_division, read_all, before_date_df, add_date, read_one


class ComputeSynergies():

    """Class to Compute Unweighted Synergy Graph from a given dataset.

    High-level walkthrough of Class:
    1. Build Random weighted, fully-connected matrix representation of graph:
        A. Gather list of matchup training data
        B. Unique players now form columns
        C. Generate the possible Edges as nChoose2
        D. From this list of edges randomly keep each with P probability
    2. Learn Player Capabilities:
        A. Find shortest distance between each player in matchup
        B. Add or subtract this distance to the index of player in the distance matrix
        C. Do this for all games in training set
        D. Normalize distance matrix number of combinations (1 / 45)
        E. Using this system of equations do least squares solver to solve Capability Matrix (C)
        F. Run Simulated Annealing until acceptable RMSE or k iterations is reached
    3. This learned Synergy Graph for predicting future matchups.

    More information on how this is created can be found in reference #1 
    (pages 22-23 specifically) and reference #3 (page 6 specifically).

    Parameters
    ----------
    df : pandas DataFrame
        DF of you want to use to build graph.
        Should contain at least these fields:
        -'i_lineup' & 'j_lineup' where each lineup row is tuple of player_ids.
        -'i_margin' & 'j_margin' where each row is integer of point 
        differentials for that respective matchup.

    Attributes
    ----------
    V : set, size (n)
        Vertices of graph representing each unique player.

    C : array, shape (n, 1)
        Capability Matrix.
        Calculated capabilities for each player.

    M : array, shape(m, n)
        Synergy Matrix.
        Mostly sparse matrix, each row = matchup, each col = different player.
        Created using 1/distance in G for each player combination in matchup.
        More information on how this is created can be found on page 6 of
        reference #3 and pages 22-23 of reference #1.

    B : array, shape(m, 1)
        Past Performance Matrix.
        Point differential for each matchup in past.

    G : networkx graph
        Unweighted Synergy Graph.
        Synergy is represented by (1 / shortest distance) between players.

    error : float
        Model error on training set. RMSE.

    C_df : pandas DataFrame
        Compatibility Matrix (C) with player names added.

    References
    ----------
    1) Modeling and Learning Synergy for Team Formation with Heterogeneous
    Agents, 2012 - Somchaya Liemhetcharat, Manuela Veloso

    2) Weighted Synergy Graphs for Effective Team Formation with
    Heterogeneous Ad Hoc Agents, 2013 - Somchaya Liemhetcharat, Manuela Veloso

    3) Adversarial Synergy Graph Model for Predicting Game Outcomes in Human
    Basketball, 2015 - Somchaya Liemhetcharat, Yicheng Luo
    """

    def __init__(self, df):
        self.df = df
        self.V = set()
        self.C = None
        self.M = None
        self.B = None
        self.G = None
        self.error = None

        self._V_index = None
        self._con = None
        self.C_df = None
        self._edge_prob = 0.5

        # self.create_graph()
        # self._get_V_and_index()
        # self.df.reset_index(drop=True, inplace=True)

    @timeit
    def create_graph(self):
        """Create the intial Synergy Graph structure from Random.

        -Get all unique players (V)
        -Get all possible Edge Combinations (E_super)
        -Randomly choose half of these (small_E)
        -Create Graph with nodes, V and edges, small_E
        """
        self.V = set()
        for row in self.df.iterrows():
            self.V |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])

        print "Starting Create Graph"
        num_verts = len(self.V)
        print num_verts

        E_super = list(combinations(self.V, 2))
        num_E_super = len(E_super)
        num_E_small = np.random.binomial(num_E_super, self._edge_prob)
        mask_E = np.random.choice(num_E_super, num_E_small, replace=False)
        small_E = np.array(E_super)[mask_E]

        self.G = nx.Graph()
        self.G.add_nodes_from(self.V)
        self.G.add_edges_from(small_E)

    def learn_capabilities(self):
        """Learn Capabilities from Unweighted Graph structure.
        -Get performance matrix, B and create empty M matrix
        -Fill in M & B by looping through each row of df and using G
        -Normalize values of M with combinations for each row, k (1/45)
        -Least Squares solution to system of equations for C. (B = M x C)
        -Compute Error
        """
        self._create_matrices()

        for lu_num in xrange(len(self.B)):
            h_lu = self.df['i_lineup'][lu_num]
            a_lu = self.df['j_lineup'][lu_num]
            self._fill_matrix(h_lu, a_lu, lu_num)
            # print lu_num / float(len(self.B))
        k = (1/sc.comb(10, 2))
        self.M = k * self.M
        self.C = np.linalg.lstsq(self.M, self.B)[0]

        self.compute_error()

    def _get_V_and_index(self):
        """Get Index for our players in V.
        We need this to know the index for each player in our M and C matrix."""
        self.V = self.G.node.keys()
        self._V_index = {}
        for ix, player in enumerate(self.V):
            self._V_index.update({player: ix})

    def _create_matrices(self):
        """Create zero M matrix where:
            row index = each lineup in training set
            column index = each player in training set
        Create B matrix from past performance (point differentials)
        """
        self.M = np.zeros((len(self.df['i_margin']), len(self.V)))
        self.B = np.array(self.df['i_margin'])
        self.B = self.B.reshape(self.B.shape[0], 1)

    def _fill_matrix(self, Ai, Aj, lu_num):
        """Fill the M matrix for a given matchup."""

        combi = list(combinations(Ai, 2))
        combj = list(combinations(Aj, 2))
        combadv = list(combinations(Ai+Aj, 2))

        for item in combi:
            combadv.remove(item)
        for item in combj:
            combadv.remove(item)

        # Loop through each combination of teammates in team i.
        # Add their pairwise synergy to M for the index of the current matchup and each respective player.
        for pair_i in combi:
            p_idx1 = self._V_index[pair_i[0]]
            p_idx2 = self._V_index[pair_i[1]]
            d = nx.shortest_path_length(self.G, pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] += 1/float(d)

        # Loop through each combination of teammates in team j
        # Subtract the pairwise synergy to M for the index of the current matchup and each respective player.
        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = nx.shortest_path_length(self.G, pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

        # Loop through each combination across team i and j.
        # Add pairwise synergy for i players and subtract for j players.
        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = nx.shortest_path_length(self.G, adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

    def compute_error(self):
        """Compute Training Error.
        RMSE between predicted point diff and actual.
        """
        pred = np.dot(self.M, self.C)
        self.error = math.sqrt(sum((pred - self.B) ** 2) / len(self.B))
        # print self.error

    def capability_matrix(self):
        """Create Capability DF that add player names to C and sorts by values."""
        self._con = connect_sql()
        C_df = pd.DataFrame(self.V, columns=['id'])
        C_df = pd.concat([C_df, pd.DataFrame(self.C, columns=['C'])], axis=1)
        p_id = pd.read_sql(sql="SELECT * from players_lookup",
                           con=self._con)
        # agg_db = pd.read_sql(
        # sql="SELECT * from agg_matchups where season ='" + season + "';",
        # con=con)

        C_df = C_df.merge(p_id, how='left', on='id')
        # C_df = C_df.merge(
        #     agg_db, how='left', left_on='id', right_on='player_id')
        # C_df.drop(['player_id', 'season'], axis=1, inplace=True)
        self.C_df = C_df.sort_values('C', ascending=False)

    @timeit
    def simulated_annealing(self, num=1000):
        """Make slight changes to initial random graph by either adding or removing one edge at a time.
        Keep graph with lowest error until there is no improvement to error for num times.

        TODO: -Add temperature component (probability of accepting worse)

        Parameters
        ----------
        num : int
            Number of times you can iterate on graph with no improvements.
        """
        try:
            num_no_improvement = 0
            while (num_no_improvement < num) & (self.error > 1):
                # print("--- %s seconds ---" % round(time.time() - start_time, 2))
                # start_time = time.time()
                print num_no_improvement, self.error
                self.old_G = self.G
                self.old_error = self.error

                rand = random.random()
                if rand > 0.5:
                    v1 = np.random.choice(self.V)
                    v2 = np.random.choice(self.V)

                    while self.G.has_edge(v1, v2) or v1 == v2:
                        v1 = np.random.choice(self.G.nodes())
                        v2 = np.random.choice(self.G.nodes())

                    self.G.add_edge(v1, v2)

                else:
                    v1 = np.random.choice(self.G.nodes())
                    v2 = np.random.choice(self.G.nodes())

                    while self.G.has_edge(v1, v2) is False or v1 == v2:
                        v1 = np.random.choice(self.G.nodes())
                        v2 = np.random.choice(self.G.nodes())

                    self.G.remove_edge(v1, v2)

                self.learn_capabilities()

                if self.error > self.old_error:
                    self.G = self.old_G
                    self.error = self.old_error
                    num_no_improvement += 1

                else:
                    num_no_improvement = 0

            return "Finished"

        except Exception, e:
            self.to_pickle()
            print e
            return "Finished with errors"

    def to_pickle(self, folder=None, name=None):
        """Pickle entire Class for future use.

        Parameters
        ----------
        folder : string, ex. "10_23"
            Directory name to store pkl.
            If none will store in 'random' directory.
        name : string, ex. "2015"
            Name of pickled class.
        """
        if name is None:
            name = ''
        if folder is None:
            folder = 'random/'
        name = '../data/cs/' + folder + '/' + name + '.pkl'
        with open(name, 'w') as f:
            pickle.dump(self, f)

        return "Pickled"

    def initialize_random_graphs(self, num=10):
        """Initialize multiple random graphs.
        Choose one with lowest error.

        This has shown in this example to greatly enhance the speed of 
        convergence if working with large dataset.
        Rather than making small changes like with simulated annealing, this 
        method can better ensure that we have not chosen a particularly bad 
        starting graph, and thus will take an extremely long time to get close 
        to convergence. 

        Parameters
        ----------
        num : int
            Number of graphs to create.
        """
        incumbent_error = 10000
        for _ in xrange(num):
            self._edge_prob = random.uniform(0.3, 0.9)
            self.create_graph()
            self._get_V_and_index()
            self.df.reset_index(drop=True, inplace=True)

            self.learn_capabilities()

            print "New:", _, self._edge_prob, self.error

            if incumbent_error > self.error:
                incumbent_graph = self.G
                incumbent_error = self.error
                incumbent_edge_prob = self._edge_prob
            else:
                self.G = incumbent_graph
                self.error = incumbent_error
                self._edge_prob = incumbent_edge_prob

            print "Old:", _, self._edge_prob, self.error

        self.learn_capabilities()


if __name__ == '__main__':
    '''Choose all, one season, or one game as df?'''
    season = '2014'
    df = read_all('matchups_reordered')
    # df = read_season('matchups_reordered', season)
    # df = add_date(df)
    # df = read_one('matchups_reordered', 'GAME_ID', '0021400008')
    train_df = add_date(df)

    '''subset on division or before given day?'''
    # df = subset_division(df, 'Pacific')
    # last_graph_day = '2015-02-26'
    # last_graph_day = datetime.strptime(last_graph_day, "%Y-%m-%d")
    # train_df = before_date_df(df, last_day=last_graph_day)

    '''Combine same matchups and remove matchups less than minute.'''
    train_df = combine_same_matchups(df)
    train_df = greater_than_minute(train_df)

    '''Compute Synergy Graph and learn Capabilities'''
    cs = ComputeSynergies(train_df)
    cs.initialize_random_graphs(10)
    # cs.simulated_annealing(1000)
    cs.learn_capabilities()
