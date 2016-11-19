import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
from combine_matchups import combine_same_matchups
from helper_functions import read_season, timeit, connect_sql


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

        self.create_graph()

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
        small_E = np.array(E_super)[mask_E]

        self.G = nx.Graph()
        self.G.add_nodes_from(self.V)
        self.G.add_edges_from(small_E)

    @timeit
    def learn_capabilities(self):
        self.V = self.G.node.keys()
        self.df.reset_index(drop=True, inplace=True)

        self._create_matrices()

        self._V_index = {}
        for ix, player in enumerate(self.V):
            self._V_index.update({player: ix})

        for lu_num in xrange(len(self.B)):
            h_lu = self.df['i_lineup'][lu_num]
            a_lu = self.df['j_lineup'][lu_num]
            self._fill_matrix(h_lu, a_lu, lu_num)
            print lu_num / float(len(self.B))
        k = (1/sc.comb(10, 2))
        self.M = k * self.M
        self.C = np.linalg.lstsq(self.M, self.B)[0]

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




    def _fill_prediction_matrix(self, Ai, Aj, lu_num):

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



    def predict_game(self, row):
        C = 


class PredictSynergy():

    def __init__(self, syn_obj):
        


    def predict_season(self, predict_df):
        players = set()
        for row in predict_df.iterrows():
            players |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])

        small_index = [self._V_index[x] for x in players]
        new_C = self.C[small_index]
        new_M = np.zeros((len(predict_df), len(players)))

        for lu_num in xrange(len(predict_df)):
            h_lu = self.df['i_lineup'][lu_num]
            a_lu = self.df['j_lineup'][lu_num]
            self._fill_prediction_matrix(h_lu, a_lu, lu_num)

if __name__ == '__main__':
    df = read_season('matchups_reordered', '2014')
    df = combine_same_matchups(df)
    cs = ComputeSynergies(df)
