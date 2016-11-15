import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx

def learn_capabilities(G, games):
	V = G.node.keys()
	# com_lus = games['home_lu'].append(games['away_lu'])
	# com_margins = games['hmargin'].append(games['amargin'])
	com_lus = games['home_lu']
	com_margins = games['hmargin']

	M, B = create_matrices(com_margins, len(V))

	runs = len(com_lus)
	for lu_num, lu in enumerate(com_lus):
		for curr_play in lu:
			 p_idx = V.index(curr_play)
			 M[lu_num, p_idx] = find_distance(G, curr_play, lu)
			 print lu_num / float(runs)
	C = np.linalg.lstsq(M, B)[0]
	err = np.sqrt(C[1] / len(com_margins))

	return V, C, err


def create_matrices(margins, num_players):
	# create M matrix
	# row index = each lineup in training set
	# column index = each player in training set
	M = np.zeros((len(margins), num_players))
	B = np.array(margins)
	B = B.reshape(B.shape[0], 1)

	return M, B


def find_distance(G, curr_play, lineup):
	d = 0.0
	lu_mates = list(lineup)
	lu_mates.remove(curr_play)

	for i in lu_mates:
		d += 1 / float(short_path_len(G, curr_play, i))
	return d


def short_path_len(G, node1, node2):
	return len(nx.shortest_path(G, node1, node2)) - 1


if __name__ == '__main__':
	games = read_season('2008')

	G = create_graph(games)

	C, err = learn_capabilities(G, games)
	np.save('first_C.npy', C)