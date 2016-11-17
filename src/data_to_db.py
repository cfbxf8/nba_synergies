import sqlalchemy
import ConfigParser
from pandas.io.json import json_normalize
import pandas as pd
import os
import json
import requests

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))


def connect_sql():
    """Connect to SQL database using a config file and sqlalchemy
    """
    try:
        ENGINE = config.get('database', 'engine')
        DATABASE = config.get('database', 'database')

        HOST = None if not config.has_option(
            'database', 'host') else config.get('database', 'host')
        USER = None if not config.has_option(
            'database', 'user') else config.get('database', 'user')
        SCHEMA = None if not config.has_option(
            'database', 'schema') else config.get('database', 'schema')
        PASSWORD = None if not config.has_option(
            'database', 'password') else config.get('database', 'password')
    except ConfigParser.NoOptionError:
        print 'Need to define engine, user, password, host, and database parameters'
        raise SystemExit

    if ENGINE == 'sqlite':
        dbString = ENGINE + ':///%s' % (DATABASE)
    else:
        if USER and PASSWORD:
            dbString = ENGINE + \
                '://%s:%s@%s/%s' % (USER, PASSWORD, HOST, DATABASE)
        elif USER:
            dbString = ENGINE + '://%s@%s/%s' % (USER, HOST, DATABASE)
        else:
            dbString = ENGINE + '://%s/%s' % (HOST, DATABASE)

    db = sqlalchemy.create_engine(dbString)
    conn = db.connect()

    return conn


def run_seasons(seasons):
    """Create SQL database from locally stored NBA stats json files.

    Loop through each season, then each game.
    Add each game to temp file (just one season), then add these to total dfs.
    Send to SQL.

    Parameters
    ----------
    seasons : list of string ex. '2014'
        Seasons to be looped through

    Returns
    -------
    "Finished" Prompt
    """

    players = pd.DataFrame()
    teams = pd.DataFrame()
    matchups = pd.DataFrame()
    team_stats = pd.DataFrame()
    starter_stats = pd.DataFrame()
    player_stats = pd.DataFrame()

    # Loop through each season.
    for season in seasons:
        file_path = '../data/raw/matchups/' + season + \
            '-' + str(int(season[-2:])+1).zfill(2)

        temp_matchups = pd.DataFrame()
        temp_team_stats = pd.DataFrame()
        temp_starter_stats = pd.DataFrame()
        temp_player_stats = pd.DataFrame()

        # Loop through each game in the season
        for filename in os.listdir(file_path):
            jso = load_one_json(season, filename)

            players = player_lookup(players, jso)
            teams = team_lookup(teams, jso)
            temp_matchups = pd.concat([temp_matchups,
                                       get_matchups(jso)])
            temp_team_stats = pd.concat([temp_team_stats,
                                         json_normalize(jso['_boxscore']['resultSets']['TeamStats'])])
            temp_starter_stats = pd.concat([temp_starter_stats,
                                            json_normalize(jso['_boxscore']['resultSets']['TeamStarterBenchStats'])])
            temp_player_stats = pd.concat([temp_player_stats,
                                           json_normalize(jso['_boxscore']['resultSets']['PlayerStats'])])

        # Add Season to each DB
        temp_matchups['season'] = season
        temp_team_stats['season'] = season
        temp_starter_stats['season'] = season
        temp_player_stats['season'] = season

        # Now concatenate each season df to master df
        matchups = pd.concat([matchups, temp_matchups]).reset_index(drop=True)
        team_stats = pd.concat(
            [team_stats, temp_team_stats]).reset_index(drop=True)
        starter_stats = pd.concat(
            [starter_stats, temp_starter_stats]).reset_index(drop=True)
        player_stats = pd.concat(
            [player_stats, temp_player_stats]).reset_index(drop=True)

    # Send to sql
    players.to_sql('players_lookup', conn, if_exists='replace', index=False)
    teams.to_sql('teams_lookup', conn, if_exists='replace', index=False)
    matchups.to_sql('matchups', conn, if_exists='replace', index=True)
    team_stats.to_sql('team_stats', conn, if_exists='replace', index=False)
    starter_stats.to_sql(
        'starter_stats', conn, if_exists='replace', index=False)
    player_stats.to_sql('player_stats', conn, if_exists='replace', index=False)

    return "Finished"


def load_one_json(season, filename):
    file_path = '../data/raw/matchups/' + season + \
        '-' + str(int(season[-2:])+1).zfill(2)
    with open(file_path + '/' + filename) as json_data:
        print filename
        jso = json.load(json_data)
    return jso


def player_lookup(players, jso):
    players = pd.concat([players,
                         json_normalize(jso['home_players'])[['id', 'name']],
                         json_normalize(jso['away_players'])[['id', 'name']]])

    return players.drop_duplicates()


def team_lookup(teams, jso):
    teams = pd.concat([teams,
                       json_normalize(jso['home_team']),
                       json_normalize(jso['away_team'])])
    return teams.drop_duplicates('id', keep='last')


def get_matchups(json_data):
    gameid = json_data['game_id']

    homeid = json_data['home_team']['id']
    awayid = json_data['away_team']['id']

    # need to check for these being reversed like on 10/28/15
    home_starters = tuple([x['id'] for x in json_data['home_starters']])
    away_starters = tuple([x['id'] for x in json_data['away_starters']])

    home_lu, away_lu = [], []
    hmargin, amargin = [], []
    htime, atime = [], []

    for i in json_data['matchups']:
        home_lu.append(tuple([x['id'] for x in i['home_players'][0]]))
        away_lu.append(tuple([x['id'] for x in i['away_players'][0]]))

        hmargin.append(i['point_difference'] * -1)
        amargin.append(i['point_difference'])

        htime.append(i['elapsed_seconds'])
        atime.append(i['elapsed_seconds'])

    temp_df = pd.DataFrame({"home_lu": home_lu, "away_lu": away_lu,
                            "hmargin": hmargin, "amargin": amargin, "htime": htime,
                            "atime": atime})

    temp_df["gameid"] = gameid
    temp_df["homeid"] = homeid
    temp_df["awayid"] = awayid

    return temp_df


def home_away_row():
    df = pd.read_sql(sql="SELECT * from team_stats", con=conn)

    game_ids = df['GAME_ID'].unique()
    home_ids = []
    away_ids = []

    headers = headers = eval(open('../headers.txt', 'r').read())

    for i in game_ids:
        url = 'http://stats.nba.com/stats/boxscoresummaryv2?GameID=' + i
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            result = r.json()['resultSets'][0]

            home_index = result['headers'].index('HOME_TEAM_ID')
            away_index = result['headers'].index('VISITOR_TEAM_ID')

            home_id = result['rowSet'][0][home_index]
            away_id = result['rowSet'][0][away_index]

            home_ids.append(home_id)
            away_ids.append(away_id)
            print i, away_id, home_id
        else:
            print r.status_code
            return r.status_code

    df = pd.DataFrame([game_ids, home_ids, away_ids]).T
    df.columns = ['game_ids', 'home_ids', 'away_ids']
    df.to_csv('output.csv')

    df.to_sql('home_away', conn, if_exists='replace', index=False)

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


if __name__ == '__main__':
    conn = connect_sql()

    seasons = [str(i) for i in range(2008, 2016)]
    run_seasons(seasons)

    # create_aggregated_db(conn)
