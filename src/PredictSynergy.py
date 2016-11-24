import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, timeit, connect_sql, subset_division


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

    # def predict_all(self):
    #     game_ids = self.orig_df['GAME_ID'].unique()
    #     for game in game_ids:
    #         self._game_id = game
    #         self.df = self.orig_df[self.orig_df['GAME_ID'] == game]
    #         self.predict_one()
    #         self.append_prediction()

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
            d = nx.shortest_path_length(self.syn_obj.G, pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] += 1/float(d)

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = nx.shortest_path_length(self.syn_obj.G, pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = nx.shortest_path_length(self.syn_obj.G, adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

    def short_path_len(self, node1, node2):
        return len(nx.shortest_path(self.syn_obj.G, node1, node2)) - 1


def predict_all(self, syn_obj, df, season):
    predict_df = pd.DataFrame()
    game_ids = df['GAME_ID'].unique()
    for game in game_ids:
        print game
        gamedf = df[df['GAME_ID'] == game].reset_index()
        ps = PredictSynergy(syn_obj, gamedf)
        try:
            ps.predict_one()
        except KeyError:
            continue
        predict_df = predict_df.append(ps.predictdf)

    predict_df.columns = ['GAME_ID', 'prediction']
    predict_df.set_index('GAME_ID', inplace=True)

    actual = read_season('matchups_reordered', season)
    actual = actual.groupby('GAME_ID').sum()['i_margin']

    com_df = pd.concat([actual, predict_df], axis=1)
    com_df['correct'] = com_df['i_margin'] * com_df['prediction'] > 0
    com_df = com_df[com_df['prediction'].notnull()]

    return com_df
