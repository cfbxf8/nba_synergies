import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
from combine_matchups import combine_same_matchups
from helper_functions import read_season, timeit, connect_sql, read_one
pd.set_option('display.width', 200)


class SynergyStepThrough():

    def __init__(self, df):
        self.df = None
        self.V = set()
        self.C = None
        self.M = None
        self.B = None
        self.G = None
        self.error = None

        self._V_index = None
        self._con = None
        self.C_df = None
        self._bigdf = df
        self._teamids = None
        self._team_M = None
        self._other_M = None

        self.create_graph()

    @timeit
    def create_graph(self):
        for row in self._bigdf.iterrows():
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

    def step_through_one_game(self):
        inpt = None
        self.player_team_id_lookup()
        for num in xrange(1, len(self._bigdf)):
            self.df = self._bigdf.iloc[0:num]
            self.learn_capabilities()

            print '*************************'
            self.compute_error()
            print self.df[['GAME_ID', 'i_margin', 'i_time']]
            self.capability_matrix()
            print '========================'
            print self.C_df

            if inpt == 'c':
                continue
            elif inpt == 'q':
                break
            inpt = raw_input()


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

        self._team_M = np.zeros((len(self.df['i_margin']), len(self.V)))
        self._other_M = np.zeros((len(self.df['i_margin']), len(self.V)))

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
            self._team_M[lu_num, p_idx1] += 1/float(d)
            self._team_M[lu_num, p_idx2] += 1/float(d)

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = self.short_path_len(pair_j[0], pair_j[1])
            self._team_M[lu_num, p_idx1] -= 1/float(d)
            self._team_M[lu_num, p_idx2] -= 1/float(d)

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = self.short_path_len(adver_pair[0], adver_pair[1])
            self._other_M[lu_num, p_idx1] += 1/float(d)
            self._other_M[lu_num, p_idx2] -= 1/float(d)

        self.M = self._team_M + self._other_M

    def short_path_len(self, node1, node2):
        return len(nx.shortest_path(self.G, node1, node2)) - 1

    def compute_error(self):
        pred = np.dot(self.M, self.C)
        self.error = np.sqrt(sum(np.exp2(pred - self.B)) / float(len(self.B)))
        print self.error

    def capability_matrix(self):
        self._con = connect_sql()
        C_df = pd.DataFrame(self.V, columns=['id'])
        C_df = pd.concat([C_df, pd.DataFrame(self.C, columns=['C'])], axis=1)
        p_id = pd.read_sql(sql="SELECT * from players_lookup",
                           con=self._con)
        C_df['C'] = C_df['C'].round(1)
        # agg_db = pd.read_sql(
        #     sql="SELECT * from agg_matchups where season ='" + season + "';", con=con)

        C_df = C_df.merge(p_id, how='left', on='id')
        C_df = C_df.merge(self._teamids[['TEAM_ABBREVIATION', 'PLAYER_ID']],
                          left_on='id', right_on='PLAYER_ID', how='left')
        C_df.drop('PLAYER_ID', axis=1, inplace=True)
        C_df = C_df.rename(columns={'TEAM_ABBREVIATION': 'Team'})
        # C_df = C_df.merge(
        #     agg_db, how='left', left_on='id', right_on='player_id')
        # C_df.drop(['player_id', 'season'], axis=1, inplace=True)

        M = pd.Series(self.M[-1], name='M').round(2)
        team_M = pd.Series(self._team_M[-1], name='team_M').round(2)
        other_M = pd.Series(self._other_M[-1], name='other_M').round(2)
        C_df = pd.concat([C_df, M, team_M, other_M], axis=1)

        i_lineup = set(self.df.iloc[-1]['i_lineup'])
        j_lineup = set(self.df.iloc[-1]['j_lineup'])
        current_lineup = i_lineup | j_lineup
        C_df['in_game'] = np.where(C_df['id'].isin(current_lineup), "in", "")

        C_df['ptd'] = 0
        cur_ptd = self.df.iloc[-1]['i_margin']
        C_df['ptd'] = np.where(C_df['id'].isin(i_lineup), C_df['ptd'] + cur_ptd, C_df['ptd'])
        C_df['ptd'] = np.where(C_df['id'].isin(j_lineup), C_df['ptd'] + cur_ptd * -1, C_df['ptd'])
        C_df['in_game'] = np.where(C_df['id'].isin(current_lineup), "in", C_df['ptd'])

        if len(self.df) > 1:
            old_lineup = set(self.df.iloc[-2]['i_lineup'] + self.df.iloc[-2]['j_lineup'])

            diff = list(old_lineup ^ current_lineup)
            mask = {}
            [mask.update({d: d in current_lineup}) for d in diff]

            C_df['change'] = ''
            for k, v in mask.iteritems():
                C_df['change'] = np.where(C_df['id'] == k, v, C_df['change'])

        self.C_df = C_df.sort_values('C', ascending=False)
        # self.C_df = C_df

    def player_team_id_lookup(self):
        self._teamids = read_one('player_stats', 'GAME_ID', '0021401224')
