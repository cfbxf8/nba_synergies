import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
import random
import time
import cPickle as pickle
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, timeit, connect_sql, subset_division


class ComputeSynergies():

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
        self._small_E = None
        self._no_edge_list = None

        self.create_graph()
        self._get_V_and_index()
        self.df.reset_index(drop=True, inplace=True)

    @timeit
    def create_graph(self):
        for row in self.df.iterrows():
            self.V |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])

        print "Starting Create Graph"
        num_verts = len(self.V)
        print num_verts

        E_super = list(combinations(self.V, 2))
        num_E_super = len(E_super)
        num_E_small = np.random.binomial(num_E_super, 0.5)
        mask_E = np.random.choice(num_E_super, num_E_small, replace=False)
        self._small_E = np.array(E_super)[mask_E]
        self._no_edge_list = np.array(E_super)[~mask_E]

        self.G = nx.Graph()
        self.G.add_nodes_from(self.V)
        self.G.add_edges_from(self._small_E)

    @timeit
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
            d = self.short_path_len(pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] += 1/float(d)

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = self.short_path_len(pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = self.short_path_len(adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

    def short_path_len(self, node1, node2):
        return len(nx.shortest_path(self.G, node1, node2)) - 1

    def compute_error(self):
        pred = np.dot(self.M, self.C)
        self.error = np.sqrt(sum(np.exp2(pred - self.B)) / len(self.B))
        print self.error

    # def read_latest(season, folder):
    #     file_path = '../data/' + folder + '/'
    #     season_filter = [x for x in os.listdir(file_path) if x[0:4] == season]
    #     last = max(season_filter)
    #     print "date:", datetime.fromtimestamp(float(last.split('_')[1].split('.')[0]))

    #     if folder == 'graphs':
    #         return nx.read_gml(file_path + last)
    #     if folder == 'capabilities':
    #         return np.load(file_path + last)

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
    def simulated_annealing(self, num):
        try:
            num_no_improvement = 0
            while (num_no_improvement < num) & (self.error > 20):
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

        except:
            self.to_pickle()
            return "Finished with errors"

    def to_pickle(self):
        with open('synergy_class.pkl', 'w') as f:
            pickle.dump(self,f)

        return "Pickled"


class PredictSynergy():

    def __init__(self, syn_obj, df):
        self.syn_obj = syn_obj
        self.orig_df = df
        self.df = df
        self.V = set()
        self.C = None
        self.M = None
        self.pred_scores = None
        self.predictdf = None

        self._V_index = None
        self._con = None
        self._game_id = self.df['GAME_ID'].iloc[0]

    def predict_all(self):
        game_ids = self.orig_df['GAME_ID'].unique()
        for game in game_ids:
            self._game_id = game
            self.df = self.orig_df[self.orig_df['GAME_ID'] == game]
            self.predict_one()
            self.append_prediction()

    def predict_one(self):
        for row in self.df.iterrows():
            self.V |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])
        self.V = list(self.V)

        self._V_index = {}
        for ix, player in enumerate(self.V):
            self._V_index.update({player: ix})

        index_of_big_C = [self.syn_obj._V_index[player] for player in self.V]

        self.C = self.syn_obj.C[index_of_big_C]
        self.M = np.zeros((len(self.df), len(self.V)))

        for lu_num in xrange(len(self.df)):
            h_lu = self.df['i_lineup'][lu_num]
            a_lu = self.df['j_lineup'][lu_num]
            self._fill_matrix(h_lu, a_lu, lu_num)

        k = (1/sc.comb(10, 2))
        self.M = k * self.M

        self.pred_scores = np.dot(self.M, self.C)
        self.predictdf = pd.DataFrame([[self._game_id, self.pred_scores.sum()]])

    def score_prediction(self):
        print self.orig_df['i_margin'].sum() * self.pred_scores.sum() > 0

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
            d = self.short_path_len(pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] += 1/float(d)

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = self.short_path_len(pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = self.short_path_len(adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

    def short_path_len(self, node1, node2):
        return len(nx.shortest_path(self.syn_obj.G, node1, node2)) - 1


def predict_all(syn_obj, df, season):
    predict_df = pd.DataFrame()
    game_ids = df['GAME_ID'].unique()
    for game in game_ids:
        print game
        gamedf = df[df['GAME_ID'] == game].reset_index()
        ps = PredictSynergy(syn_obj, gamedf)
        ps.predict_one()
        predict_df = predict_df.append(ps.predictdf)

    predict_df.columns = ['GAME_ID', 'prediction']
    predict_df.set_index('GAME_ID', inplace=True)

    actual = read_season('matchups_reordered', season)
    actual = actual.groupby('GAME_ID').sum()['i_margin']

    com_df = pd.concat([actual, predict_df], axis=1)
    com_df['correct'] = com_df['i_margin'] * com_df['prediction'] > 0

    return com_df


if __name__ == '__main__':
    season = '2014'
    df = read_season('matchups_reordered', season)
    df = subset_division(df, 'Pacific')
    df = combine_same_matchups(df)
    df = greater_than_minute(df)
    cs = ComputeSynergies(df)
    cs.learn_capabilities()
    cs.simulated_annealing(1000)

    # predictions = predict_all(cs, df, season)
