import pandas as pd
import numpy as np
from helper_functions import connect_sql, read_all


def create_reordered_matchups():
    con =connect_sql()
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

    # check for starting
    new_df['starting'] = np.where(new_df.duplicated('GAME_ID', keep='first'), 0, 1)

    new_df.to_sql('matchups_reordered', con, if_exists='replace', index=True)

    return "Finished"


def combine_matchups(df):
    df = df.groupby(['i_lineup', 'j_lineup']).sum()
    df.reset_index(inplace=True)

    if transform == 'starters':
        df = df.drop_duplicates('GAME_ID')
        df.reset_index(inplace=True)

    return df