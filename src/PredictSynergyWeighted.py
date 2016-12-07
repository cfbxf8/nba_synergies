import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, timeit, connect_sql, subset_division, add_date
from sklearn.cross_validation import train_test_split
from ComputeWeightedSynergies import ComputeWeightedSynergies


class PredictSynergyWeighted():

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

        index_of_big_C = []
        players_not_in_G = []

        for player in self.V:
            index_of_big_C.append(self.syn_obj._V_index[player])
            players_not_in_G.append(player)

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

        self.score_each_entry()

    def score_prediction(self):
        print self.orig_df['i_margin'].sum() * self.pred_scores.sum() > 0

    def score_each_entry(self):
        self.by_entry = pd.DataFrame([self.df['i_margin'], self.pred_scores])

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
            # d = nx.shortest_path_length(self.G, pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += self.syn_obj.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] += self.syn_obj.dist[p_idx1, p_idx2]

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            # d = nx.shortest_path_length(self.G, pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= self.syn_obj.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] -= self.syn_obj.dist[p_idx1, p_idx2]

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            # d = nx.shortest_path_length(self.G, adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += self.syn_obj.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] -= self.syn_obj.dist[p_idx1, p_idx2]

    # def short_path_len(self, node1, node2):
    #     return len(nx.shortest_path(self.syn_obj.G, node1, node2)) - 1


def predict_all(syn_obj, df, season):
    predict_df = pd.DataFrame()
    game_ids = df['GAME_ID'].unique()
    for game in game_ids:
        print game
        gamedf = df[df['GAME_ID'] == game].reset_index()
        ps = PredictSynergyWeighted(syn_obj, gamedf)
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

def predict_matchups(syn_obj, df, season):
    predict_df = pd.DataFrame()
    game_ids = df['GAME_ID'].unique()
    for game in game_ids:
        print game
        gamedf = df[df['GAME_ID'] == game].reset_index()
        ps = PredictSynergyWeighted(syn_obj, gamedf)
        try:
            ps.predict_one()
        except KeyError:
            continue
        predict_df = predict_df.append(ps.by_entry)

    # predict_df.columns = ['GAME_ID', 'prediction', 'starters']
    # predict_df.set_index('GAME_ID', inplace=True)

    # actual = read_all('matchups_reordered')
    # # actual = read_season('matchups_reordered', season)
    # actual = actual.groupby('GAME_ID').sum()['i_margin']

    # com_df = pd.concat([actual, predict_df], axis=1)
    # com_df['correct'] = com_df['i_margin'] * com_df['prediction'] > 0
    # com_df = com_df[com_df['prediction'].notnull()]

    return predict_df


if __name__ == '__main__':
    X = read_season('matchups_reordered', '2008')
    X = add_date(X)
    all_predictions = pd.DataFrame()
    k_folds = 3
    for _ in xrange(k_folds):
        train_df, test_df = train_test_split(X, test_size=0.1)
        train_df = combine_same_matchups(train_df)
        train_df = greater_than_minute(train_df)
        cs = ComputeWeightedSynergies(train_df)
        cs.genetic_algorithm(100)   
        predictions = predict_all(cs, test_df, '2008')
        all_predictions = pd.concat([all_predictions, predictions])
    all_predictions.to_csv('k_fold_weighted.csv')
