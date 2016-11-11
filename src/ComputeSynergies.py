import numpy as np
from itertools import combinations
import scipy.misc as sc

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
		Sj += synergy_pair_diff(d, pair_j[0], pair_j[1])

	for adver_pair in combadv:
		d = len(nx.shortest_path(G, adver_pair[0], adver_pair[1])) - 1
		Sadv += synergy_pair_diff(d, adver_pair[0], adver_pair[1])

	S = (1/sc.comb(10,2)) * (Si - Sj + Sadv)
	import pdb; pdb.set_trace()
	return S