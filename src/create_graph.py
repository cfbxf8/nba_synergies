import numpy as np
from itertools import combinations
import scipy.misc as sc

df = games.reset_index()

for row in df.iterrows():
	agents = agents + row[1]['home_lu'] + row[1]['away_lu']
	print row[0] / float(len(df))

V = set(agents)
num_verts = len(V)
E_super = list(combinations(V, 2))
num_E_super = len(E_super)
num_E_small = np.random.binomial(num_E_super, 0.5)
mask_E = np.random.choice(num_E_super, num_E_small, replace=False)
small_E = np.array(E_super)[mask_E]

G=nx.Graph()
G.add_nodes_from(V)
G.add_edges_from(small_E)
nx.draw(G)
plt.show()



# def first_element(G, tup):
# 	combos = list(combinations(tup,2))
# 	S = 0.0
# 	k = []
# 	for i in combos:
# 		k.append(1/float(short_path_len(G, i[0], i[1])))
# 	return combos, k





# for row1 in df['home_lu']:
# 	for row2 in df['away_lu']:
# 		V = list(combinations(one+two, 2))
# 		num_verts = len(V)
# 		E_super = list(combinations(V, 2))
# 		num_E_super = len(E_super)
# 		num_E_small = np.random.binomial(num_E_super, 0.5)
# 		mask_E = np.random.choice(num_E_super, num_E_small, replace=False)
# 		small_E = np.array(E_super)[mask_E]
