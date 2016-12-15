import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd


class PredictSynergy():

    """Class to Predict matchup outcomes from Unweighted Synergy Graph

    Parameters
    ----------
    syn_obj : ComputeSynergies obj
        Already solved for Synergy Graph from ComputeSynergies Class

    df : pandas DataFrame
        DF of you want to use to predict outcomes.
        Should contain at least these fields:
        -'i_lineup' & 'j_lineup' where each lineup row is tuple of player_ids.
        * If you also want to test performance then also need
        -'i_margin' & 'j_margin' where each row is integer of point
        differentials for that respective matchup.

    Attributes
    ----------
    V : set, size (n)
        Vertices of graph representing each unique player.

    C : array, shape (n, 1)
        Capability Matrix.
        Calculated capabilities for each player.

    M : array, shape (m, n)
        Synergy Matrix.
        Mostly sparse matrix, each row = matchup, each col = different player.
        Created using 1/distance in G for each player combination in matchup.
        More information on how this is created can be found on page 6 of
        reference #3 and pages 22-23 of reference #1.

    predictdf : pandas DataFrame
        DataFrame with predictions for each game.
        Predictions for both all of the matchups in the game and starters-only.

    by_entry : pandas DataFrame
        DataFrame with predictions at the more granular matchup-level.

    References
    ----------
    1) Modeling and Learning Synergy for Team Formation with Heterogeneous
    Agents, 2012 - Somchaya Liemhetcharat, Manuela Veloso

    2) Weighted Synergy Graphs for Effective Team Formation with
    Heterogeneous Ad Hoc Agents, 2013 - Somchaya Liemhetcharat, Manuela Veloso

    3) Adversarial Synergy Graph Model for Predicting Game Outcomes in Human
    Basketball, 2015 - Somchaya Liemhetcharat, Yicheng Luo
    """

    def __init__(self, syn_obj, df):
        self.syn_obj = syn_obj
        self.df = df
        self.V = set()
        self.C = None
        self.M = None
        self.predictdf = None
        self.by_entry = None

        # self._pred_scores = None
        # self._V_index = None
        # self._con = None
        self._game_id = self.df['GAME_ID'].iloc[0]

    def predict_one(self):
        """Predict one game of matchups given a Synergy Graph and player capabilities. """
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

        self._pred_scores = np.dot(self.M, self.C)
        self.predictdf = pd.DataFrame([[self._game_id, self._pred_scores.sum(), self._pred_scores[0][0]]])

        self.score_each_entry()

    def score_each_entry(self):
        """Get by_entry attribute which is the comparison of predictions to actuals for the most granular matchup case."""
        self.by_entry = pd.DataFrame(self._pred_scores)
        self.by_entry = self.by_entry.rename(columns={0: 'prediction_by_matchup'})

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
            d = nx.shortest_path_length(self.syn_obj.G, pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] += 1/float(d)

        # Loop through each combination of teammates in team j
        # Subtract the pairwise synergy to M for the index of the current matchup and each respective player.
        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = nx.shortest_path_length(self.syn_obj.G, pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)

        # Loop through each combination across team i and j.
        # Add pairwise synergy for i players and subtract for j players.
        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = nx.shortest_path_length(self.syn_obj.G, adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += 1/float(d)
            self.M[lu_num, p_idx2] -= 1/float(d)
