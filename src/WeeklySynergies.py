import pandas as pd
from datetime import datetime, timedelta
from helper_functions import read_all, read_season, before_date_df
from combine_matchups import greater_than_minute, combine_same_matchups
from ComputeSynergies import ComputeSynergies
from PredictSynergy import PredictSynergy


class WeeklySynergies():

    def __init__(self, num_test_days, folder):
        self.num_test_days = num_test_days
        self.folder = folder

        self.all_predictions = pd.DataFrame()
        self.predict_df = None
        self.capability_df = None

        self._actual_scores = None
        self._get_actual_scores()

    def run_all_seasons(self):
        seasons = [str(i) for i in range(2008, 2016)]

        for s in seasons:
            self.capability_df = pd.DataFrame()
            self.run_one_season(s)
            self.all_predictions = pd.concat([self.all_predictions, self.predict_df])

    def run_one_season(self, season):
        self.predict_df = pd.DataFrame()
        df = read_season('matchups_reordered', season)
        df.sort_values('index', inplace=True)
        df = self.add_date(df)
        self._last_graph_day = df.date.min()
        self._last_graph_day = datetime.strptime(self._last_graph_day, "%Y-%m-%d")
        last_day_of_season = df.date.max()
        last_day_of_season = datetime.strptime(last_day_of_season, "%Y-%m-%d")
        last_test_day = datetime(1, 1, 1)

        while last_test_day < datetime(2008, 11, 17):
            self._last_graph_day += timedelta(days=self.num_test_days)
            train_df = self.get_train_df(df)
            last_test_day = self._last_graph_day + timedelta(days=self.num_test_days)

            cs = self.fit_graph(train_df)

            temp_predict_df = self.predict(cs, df, last_test_day)

            self.predict_df = pd.concat([self.predict_df, temp_predict_df])

        self.predict_df = self._actual_scores.merge(self.predict_df, left_index=True, right_index=True)
        self.predict_df = self.predict_df[self.predict_df['prediction'].notnull()]
        self.predict_df['correct'] = self.predict_df['i_margin'] * self.predict_df['prediction'] > 0

        self._to_csv(season)

    def get_train_df(self, df):
        train_df = before_date_df(df, self._last_graph_day)
        train_df = combine_same_matchups(train_df)
        train_df = greater_than_minute(train_df)
        return train_df

    def fit_graph(self, train_df):
        cs = ComputeSynergies(train_df)
        cs.create_many_random_graphs(2)
        # cs.simulated_annealing(10)
        file_name = str(self._last_graph_day)[:10]
        cs.to_pickle(folder=self.folder, name=file_name)
        self.get_capabilities(cs)
        return cs

    def predict(self, cs, df, last_test_day):
        test_df = df[(df['date'] > self._last_graph_day) & (df['date'] <= last_test_day)]
        test_df = combine_same_matchups(test_df)
        test_df = greater_than_minute(test_df)

        predict_df = self.predict_over_dates(cs, test_df)
        return predict_df

    def predict_over_dates(self, syn_obj, df):
        predict_df = pd.DataFrame()
        game_ids = df['GAME_ID'].unique()
        for game in game_ids:
            print game
            gamedf = df[df['GAME_ID'] == game].reset_index()
            ps = PredictSynergy(syn_obj, gamedf)
            try:
                ps.predict_one()
            except KeyError:
                continue
            predict_df = predict_df.append(ps.predictdf)

        predict_df.columns = ['GAME_ID', 'prediction']
        predict_df.set_index('GAME_ID', inplace=True)

        predict_df['graph_day'] = str(self._last_graph_day)[:10]
        predict_df['error'] = syn_obj.error
        predict_df['edge_prob'] = syn_obj._edge_prob

        return predict_df

    def get_capabilities(self, syn_obj):
        syn_obj.capability_matrix()
        if self.capability_df is None:
            self.capability_df = syn_obj.C_df
        else:
            self.capability_df = self.capability_df.merge(syn_obj.C_df,
                                                          on=['id', 'name'],
                                                          how='outer')
        col_name = str(self._last_graph_day)[:10]
        self.capability_df = self.capability_df.rename(columns={"C": col_name})

    def add_date(self, df):
        date_df = read_all('home_away')
        df = df.merge(date_df, on='GAME_ID')
        return df

    def _get_actual_scores(self):
        actual = read_all('matchups_reordered')
        actual = actual.groupby('GAME_ID').sum()['i_margin']
        actual = pd.DataFrame(actual)
        self._actual_scores = actual

    def _to_csv(self, season):
        c_path = '../data/capabilities/' + self.folder + '/C_df_' + season + '.csv'
        self.capability_df.to_csv(c_path)

        pred_path = '../data/predictions/' + self.folder + '/pred_' + season + '.csv'
        self.predict_df.to_csv(pred_path)

    def _remove_before_storage(self):
        ''' Remove variables that you don't want to store'''


if __name__ == '__main__':
    ws = WeeklySynergies(7, folder='10_23')
    ws.run_one_season('2008')
