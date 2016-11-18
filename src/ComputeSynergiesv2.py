import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
import time
import os
from datetime import datetime
from data_to_db import connect_sql
from combine_matchups import read_season, combine_matchups


def create_graph(df):
    V = set()
    print "Starting Create Graph"
    for row in df.iterrows():
        V |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])

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
    games.reset_index(drop=True, inplace=True)
    
    lineups = games['i_lineup']
    margins = games['i_margin']

    M, B = create_matrices(margins, len(V))

    runs = len(lineups)
    for lu_num in xrange(len(games)):
        h_lu = games['i_lineup'][lu_num]
        a_lu = games['j_lineup'][lu_num]
        M = fill_matrix(G, M, V, h_lu, a_lu, lu_num)
        print lu_num / float(runs)
    k = (1/sc.comb(10, 2))
    M = k * M
    C = np.linalg.lstsq(M, B)[0]

    return V, C, M, B


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
        M[lu_num, p_idx1] -= 1/float(d)
        M[lu_num, p_idx2] -= 1/float(d)

    for adver_pair in combadv:
        p_idx1 = V.index(adver_pair[0])
        p_idx2 = V.index(adver_pair[1])
        d = len(nx.shortest_path(G, adver_pair[0], adver_pair[1])) - 1
        M[lu_num, p_idx1] += 1/float(d)
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

    S = (1/sc.comb(10, 2)) * (Si - Sj + Sadv)

    return S


def predict_season(G, games):

    predictions = []
    scores = games.groupby('gameid')['i_margin'].sum()
    starters = games[games['index'] == 0]
    for i in xrange(len(starters)):
        S = synergy_two_teams(
            G, starters['i_lineup'].iloc[i], starters['j_lineup'].iloc[i])
        predictions.append(S)
    scores = pd.concat(
        [scores.reset_index(), pd.Series(predictions)], ignore_index=True, axis=1)
    return scores


def read_latest(season, folder):
    file_path = '../data/' + folder + '/'
    season_filter = [x for x in os.listdir(file_path) if x[0:4] == season]
    last = max(season_filter)
    print "date:", datetime.fromtimestamp(float(last.split('_')[1].split('.')[0]))

    if folder == 'graphs':
        return nx.read_gml(file_path + last)
    if folder == 'capabilities':
        return np.load(file_path + last)


def capability_matrix(con, V, C, season):
    C_df = pd.DataFrame(V, columns=['id'])
    C_df = pd.concat([C_df, pd.DataFrame(C, columns=['C'])], axis=1)
    p_id = pd.read_sql(sql="SELECT * from players_lookup",
                       con=con)
    # agg_db = pd.read_sql(
    #     sql="SELECT * from agg_matchups where season ='" + season + "';", con=con)
    p_id_unique = p_id.drop_duplicates()
    com_df = C_df.merge(p_id_unique, how='left', on='id')
    # com_df = com_df.merge(
    #     agg_db, how='left', left_on='id', right_on='player_id')
    # com_df.drop(['player_id', 'season'], axis=1, inplace=True)
    com_df = com_df.sort_values('C', ascending=False)

    return com_df


def run_smaller(data, season):
    # data = read_season(season, con)
    matchups = combine_matchups(data)
    matchups = matchups.drop_duplicates()

    G = create_graph(matchups)
    V, C, M, B = learn_capabilities(G, matchups)

    err_comp = np.sqrt(sum(np.exp2(np.dot(M, C) - B)) / len(B))
    print err_comp


if __name__ == '__main__':
    season = '2014'
    con = connect_sql()

    # data = read_season(season, con)

    # matchups = combine_matchups(data, transform='starters')
    matchups = combine_matchups(data)

    G = create_graph(matchups)

    # nx.write_gml(G, '../data/graphs/'+ season + '_' + str(int(time.time())))
    # G = read_latest(season, 'graphs')

    V, C, M, B = learn_capabilities(G, matchups)

    # np.save('../data/capabilities/'+ season +'_'+ str(int(time.time())) +'.npy', C)
    # capability = read_latest(season, 'capabilities')
    err_comp = np.sqrt(sum(np.exp2(np.dot(M, C) - B)) / len(B))
    print err_comp

    # predictions = predict_season(G, games)
