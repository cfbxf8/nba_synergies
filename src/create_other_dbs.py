import pandas as pd


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