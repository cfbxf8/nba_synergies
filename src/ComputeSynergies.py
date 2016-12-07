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
# from PredictSynergy import PredictSynergy, predict_all


class ComputeSynergies():

    def __init__(self, df, graph_location=None):
        self.df = df
        self.V = set()
        self.C = None
        self.M = None
        self.B = None
        self.G = None
        self.error = None
        self.graph_location = graph_location

        self._V_index = None
        self._con = None
        self.C_df = None
        self._edge_prob = 0.5

        # self.create_graph()
        # self._get_V_and_index()
        # self.df.reset_index(drop=True, inplace=True)

    @timeit
    def create_graph(self):
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

        self.V = self.G.node.keys()
        self._V_index = {}
        for ix, player in enumerate(self.V):
            self._V_index.update({player: ix})

    def _create_matrices(self):
        # create M matrix
        # row index = each lineup in training set
        # column index = each player in training set
        self.M = np.zeros((len(self.df['i_margin']), len(self.V)))
        self.B = np.array(self.df['i_margin'])
        self.B = self.B.reshape(self.B.shape[0], 1)

    def _fill_matrix(self, Ai, Aj, lu_num):

        combi = list(combinations(Ai, 2))
        combj = list(combinations(Aj, 2))
        combadv = list(combinations(Ai+Aj, 2))

        for item in combi:
            combadv.remove(item)
        for item in combj:
            combadv.remove(item)

        for pair_i in combi:
            p_idx1 = self._V_index[pair_i[0]]
            p_idx2 = self._V_index[pair_i[1]]
            d = nx.shortest_path_length(self.G, pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] += 1/float(d)

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = nx.shortest_path_length(self.G, pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = nx.shortest_path_length(self.G, adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

    # def short_path_len(self, node1, node2):
    #     return len(nx.shortest_path(self.G, node1, node2)) - 1

    def compute_error(self):
        pred = np.dot(self.M, self.C)
        self.error = math.sqrt(sum((pred - self.B) ** 2) / len(self.B))
        # print self.error

    def capability_matrix(self):
        self._con = connect_sql()
        C_df = pd.DataFrame(self.V, columns=['id'])
        C_df = pd.concat([C_df, pd.DataFrame(self.C, columns=['C'])], axis=1)
        p_id = pd.read_sql(sql="SELECT * from players_lookup",
                           con=self._con)
        # agg_db = pd.read_sql(
        #     sql="SELECT * from agg_matchups where season ='" + season + "';", con=con)

        C_df = C_df.merge(p_id, how='left', on='id')
        # C_df = C_df.merge(
        #     agg_db, how='left', left_on='id', right_on='player_id')
        # C_df.drop(['player_id', 'season'], axis=1, inplace=True)
        self.C_df = C_df.sort_values('C', ascending=False)

    @timeit
    def simulated_annealing(self, num=1000):
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
        if name is None:
            name = ''
        if folder is None:
            folder = 'random/'
        name = '../data/cs/' + folder + '/'+ name + '.pkl'
        with open(name, 'w') as f:
            pickle.dump(self, f)

        return "Pickled"

    def create_many_random_graphs(self, num=10):
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
    season = '2014'
    df = read_all('matchups_reordered')
    # df = read_season('matchups_reordered', season)
    # df = add_date(df)
    # df = read_one('matchups_reordered', 'GAME_ID', '0021400008')
    train_df = add_date(df)

    # df = subset_division(df, 'Pacific')
    # last_graph_day = '2015-02-26'
    # last_graph_day = datetime.strptime(last_graph_day, "%Y-%m-%d")

    # train_df = before_date_df(df, last_day=last_graph_day)
    train_df = combine_same_matchups(df)
    train_df = greater_than_minute(train_df)
    cs = ComputeSynergies(train_df)
    cs.create_many_random_graphs(10)
    # cs.learn_capabilities()
    # cs.simulated_annealing(1000)

    # num_test_days = 7
    # last_test_day = last_graph_day + timedelta(days=num_test_days)

    # test_df = df[(df['date'] > last_graph_day) & (df['date'] <= last_test_day)]
    # test_df = combine_same_matchups(test_df)
    # test_df = greater_than_minute(test_df)

    predictions = predict_all(cs, train_df, season)
