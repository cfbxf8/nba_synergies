import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
from collections import defaultdict
import time
import os
from datetime import datetime
from data_to_db import connect_sql


def read_season(season):
    df = pd.read_sql(sql = "SELECT t.* from sl_master t where ", con=con_sb)


def create_graph(df):
    V = set()
    print "Starting Create Graph"
    for row in df.iterrows():
        V |= set(row[1]['home_lu']) | set(row[1]['away_lu'])

    num_verts = len(V)
    E_super = list(combinations(V, 2))
    num_E_super = len(E_super)
    num_E_small = np.random.binomial(num_E_super, 0.5)
    mask_E = np.random.choice(num_E_super, num_E_small, replace=False)
    small_E = np.array(E_super)[mask_E]

    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(small_E)

    return G


def learn_capabilities(G, games):
    V = G.node.keys()

    lineups = games['home_lu']
    margins = games['hmargin']

    M, B = create_matrices(margins, len(V))

    runs = len(lineups)
    for lu_num in xrange(len(games)):
        h_lu = games['home_lu'][lu_num]
        a_lu = games['away_lu'][lu_num]
        M = fill_matrix(G, M, V, h_lu, a_lu, lu_num)
        print lu_num / float(runs)
    k = (1/sc.comb(10,2))
    M = k * M
    C = np.linalg.lstsq(M, B)[0]
    err = np.sqrt(C[1] / len(margins))

    return V, C, err, M, B


def create_matrices(margins, num_players):
    # create M matrix
    # row index = each lineup in training set
    # column index = each player in training set
    M = np.zeros((len(margins), num_players))
    B = np.array(margins)
    B = B.reshape(B.shape[0], 1)

    return M, B


def fill_matrix(G, M, V, Ai, Aj, lu_num):

    combi = list(combinations(Ai, 2))
    combj = list(combinations(Aj, 2))
    combadv = list(combinations(Ai+Aj, 2))
    
    for item in combi:
        combadv.remove(item)
    for item in combj:
        combadv.remove(item)

    for pair_i in combi:
        p_idx1 = V.index(pair_i[0])
        p_idx2 = V.index(pair_i[1])
        d = len(nx.shortest_path(G, pair_i[0], pair_i[1])) - 1
        M[lu_num, p_idx1] += 1/float(d)
        M[lu_num, p_idx2] += 1/float(d)

    for pair_j in combj:
        p_idx1 = V.index(pair_j[0])
        p_idx2 = V.index(pair_j[1])
        d = len(nx.shortest_path(G, pair_j[0], pair_j[1])) - 1
        M[lu_num, p_idx1 ] -= 1/float(d)
        M[lu_num, p_idx2] -= 1/float(d)

    for adver_pair in combadv:
        p_idx1 = V.index(adver_pair[0])
        p_idx2 = V.index(adver_pair[1])
        d = len(nx.shortest_path(G, adver_pair[0], adver_pair[1])) - 1
        M[lu_num, p_idx1 ] += 1/float(d)
        M[lu_num, p_idx2] -= 1/float(d)

    return M


def find_distance(G, curr_play, lineup):
    d = 0.0
    lu_mates = list(lineup)
    lu_mates.remove(curr_play)

    for i in lu_mates:
        d += 1 / float(short_path_len(G, curr_play, i))
    return d


def short_path_len(G, node1, node2):
    return len(nx.shortest_path(G, node1, node2)) - 1


def synergy_pair_same(dist, Ca, Cb):
    return (1/dist) * (Ca + Cb)


def synergy_pair_diff(dist, Ca, Cb):
    return (1/dist) * (Ca - Cb)


def synergy_two_teams(G, Ai, Aj):
    Si = 0.0
    Sj = 0.0
    Sadv = 0.0

    combi = list(combinations(Ai, 2))
    combj = list(combinations(Aj, 2))
    combadv = list(combinations(Ai+Aj, 2))
    
    for item in combi:
        combadv.remove(item)
    for item in combj:
        combadv.remove(item)

    for pair_i in combi:
        d = len(nx.shortest_path(G, pair_i[0], pair_i[1])) - 1
        Si += synergy_pair_same(d, pair_i[0], pair_i[1])

    for pair_j in combj:
        d = len(nx.shortest_path(G, pair_j[0], pair_j[1])) - 1
        Sj += synergy_pair_same(d, pair_j[0], pair_j[1])

    for adver_pair in combadv:
        d = len(nx.shortest_path(G, adver_pair[0], adver_pair[1])) - 1
        Sadv += synergy_pair_diff(d, adver_pair[0], adver_pair[1])

    S = (1/sc.comb(10,2)) * (Si - Sj + Sadv)

    return S


def predict_season(G, games):
    """Computes the sample mean of the log_likelihood under a covariance model


    Parameters
    ----------
    emp_cov : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested
    
    Returns
    -------
    sample mean of the log-likelihood
    """
    predictions = []
    scores = games.groupby('gameid')['hmargin'].sum()
    starters = games[games['index'] == 0]
    for i in xrange(len(starters)):
        S = synergy_two_teams(G, starters['home_lu'].iloc[i], starters['away_lu'].iloc[i])
        predictions.append(S)
    scores = pd.concat([scores.reset_index(), pd.Series(predictions)], ignore_index=True, axis=1)
    return scores


def read_latest(season, folder):
    file_path = '../data/' + folder +'/'
    season_filter = [x for x in os.listdir(file_path) if x[0:4] == season]
    last = max(season_filter)
    print "date:", datetime.fromtimestamp(float(last.split('_')[1].split('.')[0]))
    
    if folder == 'graphs':
        return nx.read_gml(file_path + last)
    if folder == 'capabilities':
        return np.load(file_path + last)


def sql_matchup_df(conn, season):
    df = pd.read_sql(sql = 
        "SELECT * from matchups where season ='"+ season +"';"
        , con=conn)

    df['home_lu'] = df.home_lu.apply(eval)
    df['away_lu'] = df.away_lu.apply(eval)
    return df


def capability_matrix(conn, V, C, season):
    C_df = pd.DataFrame(V, columns=['id'])
    C_df = pd.concat([C_df, pd.DataFrame(C, columns=['C'])], axis=1)
    p_id = pd.read_sql(sql = 
        "SELECT * from players_lookup",
         con=conn)
    agg_db = pd.read_sql(sql = 
        "SELECT * from agg_matchups where season ='"+ season +"';"
        , con=conn)
    com_df = C_df.merge(p_id, how='left', on='id')
    com_df = com_df.merge(agg_db, how='left', left_on='id', right_on='player_id')
    com_df.drop(['player_id', 'season'], axis=1, inplace=True)
    com_df = com_df.sort_values('C', ascending=False)

    return com_df


def combine_same_lineups(games):
    games['test'] = games['homeid'] > games['awayid']


def test_right_teams():
    db = pd.read_sql(sql="SELECT * from team_stats where season= '2014'", con=conn)
    dbsmall = db[['TEAM_ID', 'GAME_ID', 'PTS']]
    dbsmall['home_away'] = np.where(dbsmall.index % 2 == 0, 'away', 'home')
    resultsdb = dbsmall.pivot(index='GAME_ID', columns= 'home_away')
    flat_results = pd.DataFrame(resultsdb.to_records())
    teams = pd.read_sql("SELECT * from teams_lookup", con=conn)
    flat_results.merge(teams, how='left', left_on="(u'TEAM_ID', 'home')", right_on='id', copy=False)


if __name__ == '__main__':
    season = '2014'
    conn = connect_sql()
    # games = read_season(season).reset_index()
    # games.to_pickle('../data/games_dfs/games_' + season)
    # games = pd.read_pickle('../data/games_dfs/games_' + season)

    matchups = sql_matchup_df(conn, season)

    match_summed = 

    G = create_graph(games)

    games_summed = 
    # nx.write_gml(G, '../data/graphs/'+ season + '_' + str(int(time.time())))
    G = read_latest(season, 'graphs')
    
    V, C, err, M, B = learn_capabilities(G, games)
    # np.save('../data/capabilities/'+ season +'_'+ str(int(time.time())) +'.npy', C)
    # capability = read_latest(season, 'capabilities')

    # predictions = predict_season(G, games)
    