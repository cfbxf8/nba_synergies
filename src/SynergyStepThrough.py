import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
import math
from helper_functions import connect_sql, read_one

pd.set_option('display.width', 200)


class SynergyStepThrough():

    """Class to visualize how the Unweighted Synergy Graph works by 
    stepping-through one game matchup by matchup and seeing how the capability 
    matrix and compute error change with changes to lineups and different point margins. 

    Simulates walking through a game from start to finish and seeing how the graph is effected.

    Shows for each player:
    - Avg. synergy to other players on court (1/shortest distance) = 'M'
    - In game currently or not = 'in_game'
    - Current total game point differential = 'ptd'
    - Time on court = 'time'
    - If they just subbed in ('True') or out ('False') = 'change'

    ==== Keyboard Actions ====
    - Press Enter to go to next matchup
    - Press 'c' + Enter to continue to rest of game
    - Press 'q' + Enter to leave quit


    Example Output:
    *************************
    Error: 0.0
          GAME_ID  i_margin  i_time
    0  0020800001         8     322
    CLE total = 8
    ========================
            id    C                name Team     M in_game  ptd  time
    9      980  6.0  Zydrunas Ilgauskas  CLE  0.18      in    8   322
    13    1112  6.0         Ben Wallace  CLE  0.18      in    8   322
    17    2590  5.2         Mo Williams  CLE  0.16      in    8   322
    0     2753  4.5        Delonte West  CLE  0.13      in    8   322
    7     2544  4.5        LeBron James  CLE  0.13      in    8   322
    5      951 -4.5           Ray Allen  BOS -0.13      in   -8   322
    11    1718 -4.8         Paul Pierce  BOS -0.14      in   -8   322
    16  200765 -4.8         Rajon Rondo  BOS -0.14      in   -8   322
    2      708 -5.2       Kevin Garnett  BOS -0.16      in   -8   322
    4     2570 -6.0    Kendrick Perkins  BOS -0.18      in   -8   322
    Sum of capabilities: 0.9

    *************************
    Error: 1.88411095042e-15
          GAME_ID  i_margin  i_time
    0  0020800001         8     322
    1  0020800001         0     101
    CLE total = 8
    ========================
            id     C                name Team     M in_game  ptd  time change
    8     2067  25.5         Eddie House  BOS -0.18      in    0   101   True
    9      980   5.8  Zydrunas Ilgauskas  CLE  0.17      in    8   423
    13    1112   4.2         Ben Wallace  CLE  0.18      in    8   423
    17    2590   3.7         Mo Williams  CLE  0.16      in    8   423
    0     2753   1.5        Delonte West  CLE  0.14      in    8   423
    7     2544   1.5        LeBron James  CLE  0.14      in    8   423
    5      951  -1.5           Ray Allen  BOS -0.14      in   -8   423
    2      708  -2.1       Kevin Garnett  BOS -0.17      in   -8   423
    11    1718  -3.4         Paul Pierce  BOS -0.14      in   -8   423
    4     2570  -4.2    Kendrick Perkins  BOS -0.18      in   -8   423
    16  200765 -24.1         Rajon Rondo  BOS  0.00           -8   322  False
    Sum of capabilities: 6.9


    Parameters
    ----------
    df : pandas DataFrame
        DF of you want to use to step-through.
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
    """
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
        self._i_abbr = None

        self.create_graph()

    def create_graph(self):
        """Create the intial Synergy Graph structure from Random.

        -Get all unique players (V)
        -Get all possible Edge Combinations (E_super)
        -Randomly choose half of these (small_E)
        -Create Graph with nodes, V and edges, small_E
        """
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
        for num in xrange(1, len(self._bigdf) + 1):
            self.df = self._bigdf.iloc[0:num]
            self.learn_capabilities()

            print '*************************'
            self.compute_error()
            print self.df[['GAME_ID', 'i_margin', 'i_time']]
            if num == 1:
                self.create_capability_matrix()
            else:
                self.update_capability_matrix()
            print self._i_abbr, "total =", self.df[['i_margin']].sum()[0]
            print '========================'
            print self.C_df[self.C_df['C'] != 0.0].sort_values('C', ascending=False)
            print "Sum of capabilities:", self.C_df['C'].sum()

            if inpt == 'c':
                continue
            elif inpt == 'q':
                break
            inpt = raw_input()

    def learn_capabilities(self):
        """Learn Capabilities from Unweighted Graph structure.
        -Get performance matrix, B and create empty M matrix
        -Fill in M & B by looping through each row of df and using G
        -Normalize values of M with combinations for each row, k (1/45)
        -Least Squares solution to system of equations for C. (B = M x C)
        -Compute Error
        """
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
        k = (1/sc.comb(10, 2))
        self.M = k * self.M
        self.C = np.linalg.lstsq(self.M, self.B)[0]

    def _create_matrices(self):
        """Create zero M matrix where:
            row index = each lineup in training set
            column index = each player in training set
        Create B matrix from past performance (point differentials)
        Also, create _team_M and _other_M for possible inclusion in output.
        """
        self.M = np.zeros((len(self.df['i_margin']), len(self.V)))
        self.B = np.array(self.df['i_margin'])
        self.B = self.B.reshape(self.B.shape[0], 1)

        self._team_M = np.zeros((len(self.df['i_margin']), len(self.V)))
        self._other_M = np.zeros((len(self.df['i_margin']), len(self.V)))

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
            self._team_M[lu_num, p_idx1] += 1/float(d)
            self._team_M[lu_num, p_idx2] += 1/float(d)

        # Loop through each combination of teammates in team j
        # Subtract the pairwise synergy to M for the index of the current matchup and each respective player.
        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            d = nx.shortest_path_length(self.G, pair_j[0], pair_j[1])
            self._team_M[lu_num, p_idx1] -= 1/float(d)
            self._team_M[lu_num, p_idx2] -= 1/float(d)

        # Loop through each combination across team i and j.
        # Add pairwise synergy for i players and subtract for j players.
        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            d = nx.shortest_path_length(self.G, adver_pair[0], adver_pair[1])
            self._other_M[lu_num, p_idx1] += 1/float(d)
            self._other_M[lu_num, p_idx2] -= 1/float(d)

        self.M = self._team_M + self._other_M

    def compute_error(self):
        """Compute Training Error.
        RMSE between predicted point diff and actual.
        """
        pred = np.dot(self.M, self.C)
        self.error = math.sqrt(sum((pred - self.B) ** 2) / len(self.B))
        print "Error:", self.error

    def create_capability_matrix(self):
        """Create Capability DF that add player names, team names, synergies, in game or not, point diff, and time for each player to C matrix and sorts by values."""
        self._con = connect_sql()
        C_df = pd.DataFrame(self.V, columns=['id'])
        C_df = pd.concat([C_df, pd.DataFrame(self.C, columns=['C'])], axis=1)
        p_id = pd.read_sql(sql="SELECT * from players_lookup",
                           con=self._con)

        # agg_db = pd.read_sql(
        #     sql="SELECT * from agg_matchups where season ='" + season + "';", con=con)

        C_df['C'] = C_df['C'].round(1)
        C_df = C_df.merge(p_id, how='left', on='id')
        C_df = C_df.merge(self._teamids[['TEAM_ABBREVIATION', 'PLAYER_ID']],
                          left_on='id', right_on='PLAYER_ID', how='left')
        C_df.drop('PLAYER_ID', axis=1, inplace=True)
        C_df = C_df.rename(columns={'TEAM_ABBREVIATION': 'Team'})

        # C_df = C_df.merge(
        #     agg_db, how='left', left_on='id', right_on='player_id')
        # C_df.drop(['player_id', 'season'], axis=1, inplace=True)

        M = pd.Series(self.M[-1], name='M').round(2)
        # team_M = pd.Series(self._team_M[-1], name='team_M').round(2)
        # other_M = pd.Series(self._other_M[-1], name='other_M').round(2)
        # C_df = pd.concat([C_df, M, team_M, other_M], axis=1)
        C_df = pd.concat([C_df, M], axis=1)

        i_lineup = set(self.df.iloc[-1]['i_lineup'])
        j_lineup = set(self.df.iloc[-1]['j_lineup'])
        cur_lineup = i_lineup | j_lineup
        C_df['in_game'] = np.where(C_df['id'].isin(cur_lineup), "in", "")

        C_df['ptd'] = 0
        cur_ptd = self.df.iloc[-1]['i_margin']
        C_df['ptd'] = np.where(C_df['id'].isin(i_lineup), cur_ptd, C_df['ptd'])
        C_df['ptd'] = np.where(C_df['id'].isin(j_lineup), cur_ptd * -1, C_df['ptd'])

        C_df['time'] = 0
        cur_time = self.df.iloc[-1]['i_time']
        C_df['time'] = np.where(C_df['id'].isin(cur_lineup), cur_time + C_df['time'], C_df['time'])

        # C_df['diff'] = 0

        self._i_abbr = self._teamids[self._teamids['TEAM_ID'] == self.df['i_id'].iloc[0]].iloc[0]['TEAM_ABBREVIATION']

        self.C_df = C_df

    def update_capability_matrix(self):
        self.C_df['C'] = self.C.round(1)
        self.C_df['M'] = pd.Series(self.M[-1], name='M').round(2)
        # self.C_df['team_M'] = pd.Series(self._team_M[-1], name='team_M').round(2)
        # self.C_df['other_M'] = pd.Series(self._other_M[-1], name='other_M').round(2)

        i_lineup = set(self.df.iloc[-1]['i_lineup'])
        j_lineup = set(self.df.iloc[-1]['j_lineup'])
        cur_lineup = i_lineup | j_lineup
        self.C_df['in_game'] = np.where(self.C_df['id'].isin(cur_lineup), "in", "")

        cur_ptd = self.df.iloc[-1]['i_margin']
        self.C_df['ptd'] = np.where(self.C_df['id'].isin(i_lineup), self.C_df['ptd'] + cur_ptd, self.C_df['ptd'])
        self.C_df['ptd'] = np.where(self.C_df['id'].isin(j_lineup), self.C_df['ptd'] + cur_ptd * -1, self.C_df['ptd'])

        old_lineup = set(self.df.iloc[-2]['i_lineup'] + self.df.iloc[-2]['j_lineup'])

        diff = list(old_lineup ^ cur_lineup)
        mask = {}
        [mask.update({d: d in cur_lineup}) for d in diff]

        self.C_df['change'] = ''
        for k, v in mask.iteritems():
            self.C_df['change'] = np.where(self.C_df['id'] == k, v, self.C_df['change'])

        cur_time = self.df.iloc[-1]['i_time']
        self.C_df['time'] = np.where(self.C_df['id'].isin(cur_lineup), cur_time + self.C_df['time'], self.C_df['time'])

        # self.C_df['diff'] = np.where(self.C_df['id'].isin(i_lineup), cur_ptd, 0)

    def player_team_id_lookup(self):
        game_id = str(self._bigdf.iloc[0]['GAME_ID'])
        self._teamids = read_one('player_stats', 'GAME_ID', game_id)


if __name__ == '__main__':
    one = read_one('matchups_reordered', 'GAME_ID', '0020800001')
    syn = SynergyStepThrough(one)
    syn.step_through_one_game()
