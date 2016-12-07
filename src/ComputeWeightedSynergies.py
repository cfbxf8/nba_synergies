import numpy as np
from itertools import combinations
import scipy.misc as sc
import networkx as nx
import pandas as pd
import random
import time
import cPickle as pickle
import math
from datetime import datetime, timedelta
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, timeit, connect_sql, subset_division, read_all, before_date_df, add_date, read_one
# from PredictSynergyWeighted import PredictSynergyWeighted, predict_all
# from pathos.multiprocessing import ProcessingPool as Pool


class ComputeWeightedSynergies():

    def __init__(self, df, high=11, graph_location=None):
        self.df = df
        self.high = high
        self.V = None
        self.C = None
        self.M = None
        self.B = None
        self.G = None
        self.error = None
        self.graph_location = graph_location
        self.dist = None

        self._V_index = None
        self._con = None
        self.C_df = None
        self._edge_prob = 0.5

        # self.create_graph()
        # self._get_V_and_index()
        # self.df.reset_index(drop=True, inplace=True)

    def create_graph(self):
        if self.V is None:
            self.create_V()

        size_V = len(self.V)
        # create random matrix
        rand_dist = np.random.randint(low=1, high=self.high, size=(size_V, size_V))
        # make symmetrical
        self.dist = self.make_symmetric(rand_dist)
        # inverse for compatability function 1/d
        self.dist = (1 / self.dist.astype(float))
        # fill diagonals as zeros, these don't matter
        np.fill_diagonal(self.dist, 0)

    @timeit
    def learn_capabilities(self):

        self._create_matrices()

        # import pdb; pdb.set_trace()
        for lu_num in xrange(len(self.B)):
            h_lu = self.df['i_lineup'][lu_num]
            a_lu = self.df['j_lineup'][lu_num]
            self._fill_matrix(h_lu, a_lu, lu_num)
            # print lu_num / float(len(self.B))
        k = (1/sc.comb(10, 2))
        self.M = k * self.M
        self.C = np.linalg.lstsq(self.M, self.B)[0]

        self.compute_error()

    def create_V(self):
        self.V = set()
        for row in self.df.iterrows():
            self.V |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])
        self._get_V_and_index()
        self.V = list(self.V)

    def _get_V_and_index(self):
        self._V_index = {}
        for ix, player in enumerate(self.V):
            self._V_index.update({player: ix})

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
            # d = nx.shortest_path_length(self.G, pair_i[0], pair_i[1])
            self.M[lu_num, p_idx1] += self.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] += self.dist[p_idx1, p_idx2]

        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            # d = nx.shortest_path_length(self.G, pair_j[0], pair_j[1])
            self.M[lu_num, p_idx1] -= self.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] -= self.dist[p_idx1, p_idx2]

        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            # d = nx.shortest_path_length(self.G, adver_pair[0], adver_pair[1])
            self.M[lu_num, p_idx1] += self.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] -= self.dist[p_idx1, p_idx2]

    # def short_path_len(self, node1, node2):
    #     return len(nx.shortest_path(self.G, node1, node2)) - 1

    def compute_error(self):
        pred = np.dot(self.M, self.C)
        self.error = math.sqrt(sum((pred - self.B) ** 2) / len(self.B))
        # print self.error

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

    @timeit
    def simulated_annealing(self, num=1000):
        try:
            num_no_improvement = 0
            while (num_no_improvement < num) & (self.error > 1):
                # print("--- %s seconds ---" % round(time.time() - start_time, 2))
                # start_time = time.time()
                print num_no_improvement, self.error
                self.old_G = self.G
                self.old_error = self.error

                rand = random.random()
                if rand > 0.5:
                    v1 = np.random.choice(self.V)
                    v2 = np.random.choice(self.V)

                    while self.G.has_edge(v1, v2) or v1 == v2:
                        v1 = np.random.choice(self.G.nodes())
                        v2 = np.random.choice(self.G.nodes())

                    self.G.add_edge(v1, v2)

                else:
                    v1 = np.random.choice(self.G.nodes())
                    v2 = np.random.choice(self.G.nodes())

                    while self.G.has_edge(v1, v2) is False or v1 == v2:
                        v1 = np.random.choice(self.G.nodes())
                        v2 = np.random.choice(self.G.nodes())

                    self.G.remove_edge(v1, v2)

                self.learn_capabilities()

                if self.error > self.old_error:
                    self.G = self.old_G
                    self.error = self.old_error
                    num_no_improvement += 1

                else:
                    num_no_improvement = 0

            return "Finished"

        except Exception, e:
            self.to_pickle()
            print e
            return "Finished with errors"

    def to_pickle(self, folder=None, name=None):
        if name is None:
            name = ''
        if folder is None:
            folder = 'random/'
        name = '../data/cs/' + folder + '/'+ name
        np.save(name, self.dist)

        return "Pickled"

    def genetic_algorithm(self, pop_size=100, cross_over_prob=.9, count=5):
        # population = Pool().map(self.initialize_population, xrange(pop_size))
        population = self.initialize_population(pop_size)
        population.sort(key=lambda x: x[0])
        self.create_V()  # have to create V because Pooling doesn't store it
        best_score = population[0][0]

        keep = int(round((1 - cross_over_prob) * pop_size))

        total_count = 0
        count_no_change = 0
        while (count > count_no_change) & (total_count < 150):
            print count_no_change

            new_population = population[0:int(keep + 1)]

            errors = [i[0] for i in population]
            while len(new_population) < 20:
                p1_idx = self.roulette_selection_min(errors)
                p1 = population[p1_idx]
                p2_idx = p1_idx
                while p2_idx == p1_idx:
                    p2_idx = self.roulette_selection_min(errors)
                    p2 = population[p2_idx]
                children = self.crossover(p1[1], p2[1])

                for child in children:
                    self.dist = child
                    self.learn_capabilities()
                    new_population.append((self.error, self.dist))
                print len(new_population), self.error
            new_population.sort(key=lambda x: x[0])
            population = new_population
            if best_score == population[0][0]:
                count_no_change += 1
                total_count += 1
            else:
                best_score = population[0][0]
                count_no_change = 0
                total_count += 1
            print best_score
            print total_count


        self.dist = population[0][1]
        self.learn_capabilities()
        self.capability_matrix()
        # self.to_pickle(name='all_season_weighted')

        return "Finished!"

    @timeit
    def initialize_population(self, pop_size):
        # self.create_graph()
        # self.learn_capabilities()
        # print self.error
        # population = (self.error, self.dist)
        # return population

        population = []
        for _ in xrange(pop_size):
            self.create_graph()
            self.learn_capabilities()
            print _, self.error
            population.append((self.error, self.dist))
        return population

    def crossover(self, father, mother):
        '''https://gist.github.com/bellbind/741853'''
        index1 = random.randint(1, father.shape[0] - 2)
        index2 = random.randint(1, father.shape[0] - 2)
        if index1 > index2:
            index1, index2 = index2, index1
        child1 = np.concatenate([father[:index1], mother[index1:index2],
                                father[index2:]])
        child2 = np.concatenate([mother[:index1], father[index1:index2],
                                mother[index2:]])
        child1 = self.make_symmetric(child1)
        child2 = self.make_symmetric(child2)
        return (child1, child2)

    def roulette_selection_min(self, errors):
        # import pdb; pdb.set_trace()
        sum_e = sum(errors)
        max_e = max(errors)
        min_e = min(errors)
        p = random.random() * sum_e
        t = max_e + min_e
        chosen_index = 0
        for idx, e in enumerate(errors):
            p -= (t - e)
            if p < 0:
                chosen_index = idx
                break
        return chosen_index

    def make_symmetric(self, child):
        new_child = child.copy()
        for col in xrange(new_child.shape[0] - 1):
            for row in xrange(col + 1, new_child.shape[0]):
                new_child[row, col] = child[col, row]
        return new_child


if __name__ == '__main__':
    season = '2014'
    # df = read_all('matchups_reordered')
    df = read_season('matchups_reordered', season)
    # df = read_one('matchups_reordered', 'GAME_ID', '0021400008')
    train_df = add_date(df)

    # df = subset_division(df, 'Pacific')
    # last_graph_day = '2015-02-26'
    # last_graph_day = datetime.strptime(last_graph_day, "%Y-%m-%d")

    # train_df = before_date_df(train_df, last_day=last_graph_day)
    train_df = combine_same_matchups(train_df)
    train_df = greater_than_minute(train_df)
    cs = ComputeWeightedSynergies(train_df)
    # cs.create_many_random_graphs(10)
    # cs.learn_capabilities()
    # cs.simulated_annealing(10)
    # multiprocesssing.Pool(7)
    # pool.map(cs.genetic_algorithm()
    cs.genetic_algorithm(pop_size=10)

    # num_test_days = 7
    # last_test_day = last_graph_day + timedelta(days=num_test_days)

    # test_df = df[(df['date'] > last_graph_day) & (df['date'] <= last_test_day)]
    # test_df = combine_same_matchups(test_df)
    # test_df = greater_than_minute(test_df)

    # predictions = predict_all(cs, test_df, season)
