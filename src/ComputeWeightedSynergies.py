import numpy as np
from itertools import combinations
import scipy.misc as sc
import pandas as pd
import random
import time
import cPickle as pickle
import math
from datetime import datetime, timedelta
from combine_matchups import combine_same_matchups, greater_than_minute
from helper_functions import read_season, timeit, connect_sql, subset_division, read_all, before_date_df, add_date, read_one


class ComputeWeightedSynergies():

    """Class to Compute Weighted Synergy Graph from a given dataset.

    High-level walkthrough of Class:
    1. Build Random weighted, fully-connected matrix representation of graph:
        A. Gather list of matchup training data
        B. Unique players now form columns
        C. Generate the edge weights as randint(1,10) for every cell
        D. Make sure graph is symmetrical
    2. Learn Player Capabilities:
        A. Find distance between each player in matchup from the matrix representation of graph
        B. Add or subtract this distance to the index of player in the distance matrix
        C. Do this for all games in training set
        D. Normalize distance matrix number of combinations (1 / 45)
        E. Using this system of equations do least squares solver to solve Capability Matrix (C)
        F. Run Genetic Algorithm until acceptable RMSE or k iterations is reached
    3. This learned Synergy Graph for predicting future matchups.

    More information on how this is created can be found in reference #1 
    (pages 22-23 specifically) and reference #3 (page 6 specifically).

    Parameters
    ----------
    df : pandas DataFrame
        DF of you want to use to build graph.
        Should contain at least these fields:
        -'i_lineup' & 'j_lineup' where each lineup row is tuple of player_ids.
        -'i_margin' & 'j_margin' where each row is integer of point 
        differentials for that respective matchup.

    high : integer > 1
        Highest number that the Weighted edges in the Synergy Graph can take.

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

    B : array, shape (m, 1)
        Past Performance Matrix.
        Point differential for each matchup in past.

    dist : array, shape (n, n)
        Matrix representation of Weighted Synergy Graph.
        The benefit of this method over creating the actual graph is that this
        table allows us to lookup the synergy of pairwise players, rather than
        having to compute shortest distance in the graph constantly.
        Once finished, can use this to build the actual representation of the graph.

    error : float
        Model error on training set. RMSE.

    C_df : pandas DataFrame
        Compatibility Matrix (C) with player names added.

    References
    ----------
    1) Modeling and Learning Synergy for Team Formation with Heterogeneous
    Agents, 2012 - Somchaya Liemhetcharat, Manuela Veloso

    2) Weighted Synergy Graphs for Effective Team Formation with
    Heterogeneous Ad Hoc Agents, 2013 - Somchaya Liemhetcharat, Manuela Veloso

    3) Adversarial Synergy Graph Model for Predicting Game Outcomes in Human
    Basketball, 2015 - Somchaya Liemhetcharat, Yicheng Luo
    """

    def __init__(self, df, high=10):
        self.df = df
        self.high = high
        self.V = None
        self.C = None
        self.M = None
        self.B = None
        self.error = None
        self.dist = None

        self._V_index = None
        self._con = None
        self.C_df = None
        self._edge_prob = 0.5

        # self.create_graph()
        # self._get_V_and_index()
        # self.df.reset_index(drop=True, inplace=True)

    def create_graph(self):
        """Create Initial Weighted Synergy Graph.
        Specifics can be seen in inline comments below.
        """
        if self.V is None:
            self.create_V()

        size_V = len(self.V)
        # create random matrix
        rand_dist = np.random.randint(low=1, high=self.high+1, size=(size_V, size_V))
        # make symmetrical
        self.dist = self.make_symmetric(rand_dist)
        # inverse for compatability function 1/d
        self.dist = (1 / self.dist.astype(float))
        # fill diagonals as zeros, as these don't have any use
        np.fill_diagonal(self.dist, 0)

    def learn_capabilities(self):
        """Learn Capabilities from Weighted Graph structure.
        -Get performance matrix, B and create empty M matrix
        -Fill in M & B by looping through each row of df and using G
        -Normalize values of M with combinations for each row, k (1/45)
        -Least Squares solution to system of equations for C. (B = M dot C)
        -Compute Error
        """
        self._create_matrices()

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
        """Create V by only getting unique players as set.
        Then convert to list."""
        self.V = set()
        for row in self.df.iterrows():
            self.V |= set(row[1]['i_lineup']) | set(row[1]['j_lineup'])
        self._get_V_and_index()
        self.V = list(self.V)

    def _get_V_and_index(self):
        """Get Index for our players in V.
        We need this to know the index for each player in our M and C matrix."""
        self._V_index = {}
        for ix, player in enumerate(self.V):
            self._V_index.update({player: ix})

    def _create_matrices(self):
        """Create zero M matrix where:
            row index = each lineup in training set
            column index = each player in training set
        Create B matrix from past performance (point differentials)
        """
        self.M = np.zeros((len(self.df['i_margin']), len(self.V)))
        self.B = np.array(self.df['i_margin'])
        self.B = self.B.reshape(self.B.shape[0], 1)

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
            self.M[lu_num, p_idx1] += self.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] += self.dist[p_idx1, p_idx2]

        # Loop through each combination of teammates in team j
        # Subtract the pairwise synergy to M for the index of the current matchup and each respective player.
        for pair_j in combj:
            p_idx1 = self._V_index[pair_j[0]]
            p_idx2 = self._V_index[pair_j[1]]
            self.M[lu_num, p_idx1] -= self.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] -= self.dist[p_idx1, p_idx2]

        # Loop through each combination across team i and j.
        # Add pairwise synergy for i players and subtract for j players.
        for adver_pair in combadv:
            p_idx1 = self._V_index[adver_pair[0]]
            p_idx2 = self._V_index[adver_pair[1]]
            self.M[lu_num, p_idx1] += self.dist[p_idx1, p_idx2]
            self.M[lu_num, p_idx2] -= self.dist[p_idx1, p_idx2]

    def compute_error(self):
        """Compute Training Error.
        RMSE between predicted point diff and actual.
        """
        pred = np.dot(self.M, self.C)
        self.error = math.sqrt(sum((pred - self.B) ** 2) / len(self.B))
        # print self.error

    def capability_matrix(self):
        """Create Capability DF that add player names to C and sorts by values."""
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

    def to_pickle(self, folder=None, name=None):
        """Save the distance matrix used for future use.

        Parameters
        ----------
        folder : string, ex. "10_23"
            directory name to store pkl.
            If none will store in 'random' directory.
        name : string, ex. "2015"
            Name of numpy file.
        """
        if name is None:
            name = ''
        if folder is None:
            folder = 'random/'
        name = '../data/cs/' + folder + '/' + name
        np.save(name, self.dist)

        return "Pickled"

    def genetic_algorithm(self, pop_size=100, cross_over_prob=.9, count=5):
        """Run a Genetic Algorithm Optimization in order more quickly converge to near the Global Minimimum compuatation error based on the Synergy Graph, the Compatibility Matrix, and Past performance of teams.
        Namely the equation B = M dot C, trying to minimimize B - pred(B)

        High Level:
        1) First generation population of chromosomes are created
            - where chromosomes = distinct randomly generated dist matrices
        2) Rank population on computed error (lowest to highest)
        3) Keep top pop_size * cross_over_prob chromosomes for next generation
            - These are exact copies (no crossover)
        4) Generate rest of next generation until you reach pop_size:
            A. Using Roulette Selection select 2 parents
                - Roulette Selection = (higher rank, higher prob of selection)
            B. Simulate breeding by crossing over chromosomes
                - Pick random row in dist matrix
                - One child has parent1 values before this point, and p2 after
                - Other child has p2 values before this point, p1 values after
            C. Make these matrices symmetrical
        5) Continue until the best score in the generation does not change for count times

        For a more in-depth explananation of the general algorithm:
            - http://www.obitko.com/tutorials/genetic-algorithms/

        Parameters
        ----------
        pop_size : int
            how many chromosomes in one generation of population
            where chromosome pop = # of dist matrices to choose from
        cross_over_prob : float, 0 to 1
            How often crossover will be performed
            where 0 = children are exact copies of old population chromosomes
        count : int
            Stopping condition (ie. continue breeding new generations until..)
            How many generations in a row where the best score is unchanged

        TODO
        ----------
        Add Mutation probability.
            - Should help if getting stuck in local minima
        """
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
            while len(new_population) < pop_size:
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
        """Create initial population with size pop_size."""
        population = []
        for _ in xrange(pop_size):
            self.create_graph()
            self.learn_capabilities()
            print _, self.error
            population.append((self.error, self.dist))
        return population

    def crossover(self, father, mother):
        """Create 2 new members of generation by combining 2 past chromosomes
        Can be thought of as simulated breeding.
        This method uses two point crossover with randomly generated points.

        Parameters
        ----------
        father : array, shape (n, n)
            dist matrix that will act as the father for the 2 children
        mother : array, shape (n, n)
            dist matrix that will act as the mother for the 2 children

        Returns
        ----------
        child1 : array, shape (n, n)
            new dist matrix for new generation
            created by crossing over mother and father chromosomes
        child2 : array, shape (n, n)
            new dist matrix for new generation
            created by crossing over mother and father chromosomes

        References
        ----------
        https://gist.github.com/bellbind/741853
        """
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
        """Select a future parent from a given list of errors
        Since this is a minimum optimization:
            - lower errors = higher prob (ie bigger area on a roulette wheel)

        For more information of the general algorithm in the max case:
            - http://www.obitko.com/tutorials/genetic-algorithms/selection.php

        Parameters
        ----------
        errors : list of floats
            computed error for each member of past population

        Returns
        ----------
        chosen_index : int, 0 to len(errors)
            Index of past population to be chosen as future parent

        References
        ----------
        http://stackoverflow.com/questions/8760473/roulette-wheel-selection-for-function-minimization
        """
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
        """Make a square matrix symmetric.

        Parameters
        ----------
        child : array, shape (n, n)
            child matrix that needs to be made symmetric

        Returns
        ----------
        new_child : array, shape (n, n)
            new symmetric using top right triangle of original matrix.
        """
        new_child = child.copy()
        for col in xrange(new_child.shape[0] - 1):
            for row in xrange(col + 1, new_child.shape[0]):
                new_child[row, col] = child[col, row]
        return new_child


if __name__ == '__main__':
    '''Choose all, one season, or one game as df?'''
    season = '2014'
    # df = read_all('matchups_reordered')
    df = read_season('matchups_reordered', season)
    # df = read_one('matchups_reordered', 'GAME_ID', '0021400008')
    train_df = add_date(df)

    '''Subset on division or before given day?'''
    # df = subset_division(df, 'Pacific')
    # last_graph_day = '2015-02-26'
    # last_graph_day = datetime.strptime(last_graph_day, "%Y-%m-%d")
    # train_df = before_date_df(train_df, last_day=last_graph_day)

    '''Combine same matchups and remove matchups less than minute.'''
    train_df = combine_same_matchups(train_df)
    train_df = greater_than_minute(train_df)

    '''Compute Synergy Graph and learn Capabilities'''
    cs = ComputeWeightedSynergies(train_df)
    cs.genetic_algorithm(pop_size=10)
    cs.learn_capabilities()
