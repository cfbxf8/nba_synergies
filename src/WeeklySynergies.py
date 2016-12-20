import pandas as pd
from datetime import datetime, timedelta
from helper_functions import read_all, read_season, before_date_df, add_date
from combine_matchups import greater_than_minute, combine_same_matchups
# from ComputeSynergies import ComputeSynergies
# from PredictSynergy import PredictSynergy
from ComputeWeightedSynergies import ComputeWeightedSynergies
from PredictSynergyWeighted import PredictSynergyWeighted


class WeeklyPredictions():

    """Class to Predict full game outcomes using Synergy Graphs.
    Predicts every week of the NBA season using a graph only game data prior to that week.
    Designed to simulate a real-life use case and determine the methods efficacy for use in predicting game outcomes.
    A more realistic cross-validation method.

    Parameters
    ----------
    folder : string
        Directory name you want to store the predictions csv, capabilities csv, and pickled files.

    num_test_days : int
        number of days to predict. default = 7 (week).
        For example, if 7 then will learn Synergy Graph on a date 
        (_last_graph_day) and then predict on next week (_last_graph_day+7).
        It will then learn Synergy Graph on (_last_graph_day + 7) and then 
        predict on next week from there, continuing until end of season.

    Attributes
    ----------
    all_predictions : Pandas DataFrame
        DataFrame with all of the appended predictions.

    capability_df : Pandas DataFrame
        Capability Matrix for given season.
    """

    def __init__(self, folder, num_test_days=7):
        self.num_test_days = num_test_days
        self.folder = folder

        self.all_predictions = pd.DataFrame()
        self.capability_df = None

        self._season_df = None
        self._actual_scores = None
        self._get_actual_scores()

    def run_all_seasons(self):
        """Run predictions across all seasons starting from 2008.
        Append each season to self.all_predictions."""
        seasons = [str(i) for i in range(2008, 2016)]

        for s in seasons:
            self.capability_df = pd.DataFrame()
            self.run_one_season(s)
            self.all_predictions = pd.concat([self.all_predictions, self._season_df])

    def run_one_season(self, season, last_day=None):
        """Run predictions for one season stepping through each week.
        Learns on days up to that point in time (training set) and then predicts over upcoming week (test set)."""
        self._season_df = pd.DataFrame()
        df = read_season('matchups_reordered', season)
        df.sort_values('index', inplace=True)
        df = add_date(df)
        self._last_graph_day = last_day
        self._last_graph_day = datetime.strptime(self._last_graph_day, "%Y-%m-%d")
        last_day_of_season = df.date.max()
        last_day_of_season = datetime.strptime(last_day_of_season, "%Y-%m-%d")
        last_test_day = datetime(1, 1, 1)

        while last_test_day < last_day_of_season:
            self._last_graph_day += timedelta(days=self.num_test_days)
            print self._last_graph_day
            train_df = self.get_train_df(df)
            last_test_day = self._last_graph_day + timedelta(days=self.num_test_days)

            cs = self.fit_graph(train_df)

            temp_predict_df = self.predict(cs, df, last_test_day)

            self._season_df = pd.concat([self._season_df, temp_predict_df])

        self._season_df = self._actual_scores.merge(self._season_df, left_index=True, right_index=True)
        self._season_df = self._season_df[self._season_df['prediction'].notnull()]
        self._season_df['match_correct'] = self._season_df['i_margin'] * self._season_df['prediction'] > 0
        self._season_df['start_correct'] = self._season_df['i_margin'] * self._season_df['start_pred'] > 0

        self._to_csv(season)

    def get_train_df(self, df):
        """Subset the training set on days before a given day."""
        train_df = before_date_df(df, self._last_graph_day)
        train_df = combine_same_matchups(train_df)
        train_df = greater_than_minute(train_df)
        return train_df

    def fit_graph(self, train_df):
        """Fit a graph with given training set."""
        ws = ComputeWeightedSynergies(train_df)
        ws.genetic_algorithm()
        # cs.simulated_annealing(10)
        file_name = str(self._last_graph_day)[:10]
        ws.to_pickle(folder=self.folder, name=file_name)
        self.get_capabilities(ws)
        return ws

    def predict(self, cs, df, last_test_day):
        """Predict outcomes given a Synergy Graph and last day to predict."""
        test_df = df[(df['date'] > self._last_graph_day) & (df['date'] <= last_test_day)]
        test_df = combine_same_matchups(test_df)
        test_df = greater_than_minute(test_df)

        predict_df = self.predict_over_dates(cs, test_df)
        return predict_df

    def predict_over_dates(self, syn_obj, df):
        """Predict outcomes given a Synergy Graph and date range to predict."""
        predict_df = pd.DataFrame()
        start_df = pd.DataFrame()
        game_ids = df['GAME_ID'].unique()
        for game in game_ids:
            print game
            gamedf = df[df['GAME_ID'] == game].reset_index()
            startdf = gamedf[gamedf.starting >= 1].reset_index()
            ps = PredictSynergyWeighted(syn_obj, gamedf)
            if len(startdf) > 0:
                ps_start = PredictSynergyWeighted(syn_obj, startdf)
                try:
                    ps_start.predict_one()
                except KeyError:
                    continue
            try:
                ps.predict_one()
            except KeyError:
                continue
            predict_df = predict_df.append(ps.predictdf)
            start_df = start_df.append(ps_start.predictdf)

        predict_df.columns = ['GAME_ID', 'prediction']
        predict_df.set_index('GAME_ID', inplace=True)

        start_df.columns = ['GAME_ID', 'start_pred']
        start_df.set_index('GAME_ID', inplace=True)

        predict_df = predict_df.merge(start_df, left_index=True,
                                      right_index=True)

        predict_df['graph_day'] = str(self._last_graph_day)[:10]
        predict_df['error'] = syn_obj.error
        predict_df['edge_prob'] = syn_obj._edge_prob

        return predict_df

    def get_capabilities(self, syn_obj):
        """Get player capability matrix."""
        syn_obj.capability_matrix()
        if self.capability_df is None:
            self.capability_df = syn_obj.C_df
        else:
            self.capability_df = self.capability_df.merge(syn_obj.C_df,
                                                          on=['id', 'name'],
                                                          how='outer')
        col_name = str(self._last_graph_day)[:10]
        self.capability_df = self.capability_df.rename(columns={"C": col_name})

    def _get_actual_scores(self):
        """Get the actual scores that occured in the games."""
        actual = read_all('matchups_reordered')
        actual = actual.groupby('GAME_ID').sum()['i_margin']
        actual = pd.DataFrame(actual)
        self._actual_scores = actual

    def _to_csv(self, season):
        """Send capability and season predictions to csv."""
        c_path = '../data/capabilities/' + self.folder + '/C_df_' + season + '.csv'
        self.capability_df.to_csv(c_path)

        pred_path = '../data/predictions/' + self.folder + '/pred_' + season + '.csv'
        self._season_df.to_csv(pred_path)


if __name__ == '__main__':
    ws = WeeklyPredictions(folder='most_recent', num_test_days=7)
    # ws.run_all_seasons()
    season = '2010'
    # season = raw_input("What Season?")
    ws.run_one_season(season)
