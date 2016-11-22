import pandas as pd
import numpy as np
from helper_functions import connect_sql, read_all


def create_reordered_matchups():
    con = connect_sql()
    df = read_all('matchups')

    column_names = {'home_lineup': 'i_lineup', 'away_lineup': 'j_lineup',
                    'home_margin': 'i_margin', 'away_margin': 'j_margin',
                    'home_time': 'i_time', 'away_time': 'j_time',
                    'home_id': 'i_id', 'away_id': 'j_id'}

    df = df.rename(columns=column_names)

    mask = df['i_id'] > df['j_id']
    true_df = df[mask]
    false_df = df[~mask]

    switch_order = {'i_lineup': 'j_lineup', 'j_lineup': 'i_lineup',
                    'i_margin': 'j_margin', 'j_margin': 'i_margin',
                    'i_time': 'j_time', 'j_time': 'i_time',
                    'i_id': 'j_id', 'j_id': 'i_id'}

    false_df = false_df.rename(columns=switch_order)

    new_df = pd.concat([true_df, false_df])

    new_df['starting'] = np.where(new_df.duplicated('GAME_ID', keep='first'), 0, 1)

    new_df.to_sql('matchups_reordered', con, if_exists='replace', index=True)

    return "Finished"


def create_aggregated_matchups(df):
    game_ids = df['GAME_ID'].unique()
    all_combined = pd.DataFrame()
    for game in game_ids:
        print game
        game_df = df[df['GAME_ID'] == game]
        one_combined = combine_same_matchups(game_df)
        all_combined = pd.concat([all_combined, one_combined], axis=1)
    return all_combined


def combine_same_matchups(df):
    sumdf = df.groupby(['i_lineup', 'j_lineup']).sum()
    sumdf = sumdf[['i_margin', 'i_time', 'j_margin', 'j_time', 'starting']]

    firstdf = df.groupby(['i_lineup', 'j_lineup']).first()[['GAME_ID', 'i_id', 'j_id', 'season', 'was_home_correct']]

    countdf = pd.DataFrame(df.groupby(['i_lineup', 'j_lineup']).count()['GAME_ID'])
    countdf.columns = ['count']

    new_df = pd.concat([sumdf, firstdf, countdf], axis=1)

    new_df.sort_values(['GAME_ID', 'i_time'], ascending=[True, False], inplace=True)

    new_df.reset_index(inplace=True)

    return new_df


def greater_than_minute(df):
    small_df = df[df['j_time'] >= 60]
    return small_df


def add_divisions():
    con = connect_sql()
    team_divisions = pd.read_csv('../data/Divisions.csv')
    team_divisions.to_sql('teams_lookup', con,
                          if_exists='replace', index=False)
    return "Finished"


def create_aggregated_db(conn):
    df = pd.read_sql(sql="SELECT * from matchups", con=conn)
    df_new = df[['amargin', 'atime', 'away_lu', 'hmargin',
                 'home_lu', 'htime', 'season']]

    df_away = df_new[['amargin', 'atime', 'away_lu', 'season']]
    df_away['away_lu'] = df_away.away_lu.apply(eval)
    df_away_new = pd.DataFrame(df_away.away_lu.tolist())
    df_away_new = pd.concat([df_away, df_away_new], axis=1)
    df_away_new = pd.melt(df_away_new, id_vars=[
                          'amargin', 'atime', 'away_lu', 'season'],
                          value_vars=[0, 1, 2, 3, 4])
    df_away_new.columns = [
        u'margin', u'time', u'lu', u'season', u'variable', u'player_id']
    df_away_new.drop(['lu', 'variable'], inplace=True, axis=1)

    df_home = df_new[['hmargin', 'htime', 'home_lu', 'season']]
    df_home['home_lu'] = df_home.home_lu.apply(eval)
    df_home_new = pd.DataFrame(df_home.home_lu.tolist())
    df_home_new = pd.concat([df_home, df_home_new], axis=1)
    df_home_new = pd.melt(df_home_new, id_vars=[
                          'hmargin', 'htime', 'home_lu', 'season'], value_vars=[0, 1, 2, 3, 4])
    df_home_new.columns = [
        u'margin', u'time', u'lu', u'season', u'variable', u'player_id']
    df_home_new.drop(['lu', 'variable'], inplace=True, axis=1)

    agg_df = pd.concat([df_home_new, df_away_new])
    agg_df.dropna(inplace=True)
    agg_df['player_id'] = agg_df['player_id'].astype(int)

    output_df = agg_df.groupby(['player_id', 'season']).sum()
    output_df_count = agg_df.groupby(['player_id', 'season']).count()

    output_df = pd.concat(
        [output_df, output_df_count['margin'].rename('count')], axis=1)

    output_df.to_sql('agg_matchups', conn, if_exists='replace', index=True)

    return "Finished"
