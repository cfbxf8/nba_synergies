import sqlalchemy
import ConfigParser
from pandas.io.json import json_normalize
import pandas as pd
import os
import json

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))

def connect_sql(config):
    try:
        ENGINE = config.get('database', 'engine')
        DATABASE = config.get('database', 'database')

        HOST = None if not config.has_option('database', 'host') else config.get('database', 'host')
        USER = None if not config.has_option('database', 'user') else config.get('database', 'user')
        SCHEMA = None if not config.has_option('database', 'schema') else config.get('database', 'schema')
        PASSWORD = None if not config.has_option('database', 'password') else config.get('database', 'password')
    except ConfigParser.NoOptionError:
        print 'Need to define engine, user, password, host, and database parameters'
        raise SystemExit

    if ENGINE == 'sqlite':
        dbString = ENGINE + ':///%s' % (DATABASE)
    else:
        if USER and PASSWORD:
            dbString = ENGINE + '://%s:%s@%s/%s' % (USER, PASSWORD, HOST, DATABASE)
        elif USER:
            dbString = ENGINE + '://%s@%s/%s' % (USER, HOST, DATABASE)
        else:
            dbString = ENGINE + '://%s/%s' % (HOST, DATABASE)
        
    db = sqlalchemy.create_engine(dbString)
    conn = db.connect()

    return conn


def run_seasons(seasons):
    all_games = pd.DataFrame()
    for season in seasons:
        file_path = '../data/raw/matchups/' + season + '-' + str(int(season[-2:])+1).zfill(2)

        players = pd.DataFrame()
        teams = pd.DataFrame()
        temp_games = pd.DataFrame()
        team_stats = pd.DataFrame()
        starter_stats = pd.DataFrame()
        player_stats = pd.DataFrame()

        for filename in os.listdir(file_path):
            with open(file_path + '/' + filename) as json_data:
                print filename
                jso = json.load(json_data)
                
            players = player_lookup(players, jso)
            teams = team_lookup(teams, jso)
            temp_games = pd.concat([temp_games, get_matchups(jso)])
            team_stats = pd.concat([team_stats, json_normalize(jso['_boxscore']['resultSets']['TeamStats'])])
            starter_stats = pd.concat([starter_stats, json_normalize(jso['_boxscore']['resultSets']['TeamStarterBenchStats'])])
            player_stats = pd.concat([player_stats, json_normalize(jso['_boxscore']['resultSets']['PlayerStats'])])
        temp_games['season'] = season
        all_games = pd.concat([all_games, temp_games]).reset_index(drop=True)


    players.to_sql('players_lookup', conn, if_exists='replace', index=False)
    teams.to_sql('teams_lookup', conn, if_exists='replace', index=False)
    all_games.to_sql('matchups', conn, if_exists='replace', index=True)
    team_stats.to_sql('team_stats', conn, if_exists='replace', index=False)
    starter_stats.to_sql('starter_stats', conn, if_exists='replace', index=False)
    player_stats.to_sql('player_stats', conn, if_exists='replace', index=False)
    return "Finished"


def player_lookup(players, jso):
    players = pd.concat([players, json_normalize(jso['home_players'])[['id', 'name']], json_normalize(jso['away_players'])[['id', 'name']]], axis=0)
    
    return players.drop_duplicates()


def team_lookup(teams, jso):
    teams = pd.concat([teams, json_normalize(jso['home_team'])])
    return teams.drop_duplicates()


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
    
    temp_df = pd.DataFrame({"home_lu":home_lu, "away_lu":away_lu, 
        "hmargin":hmargin, "amargin":amargin, "htime":htime, 
        "atime":atime})
    
    temp_df["gameid"] = gameid
    temp_df["homeid"] = homeid
    temp_df["awayid"] = awayid

    return temp_df

def create_aggregated_db(conn):
    df = pd.read_sql(sql = "SELECT * from matchups", con=conn)
    df_new = df[['amargin', 'atime', 'away_lu', 'hmargin', 'home_lu', 'htime', 'season']]
    df_away = df_new[['amargin', 'atime', 'away_lu', 'season']]
    df_away['away_lu'] = df_away.away_lu.apply(eval)
    df_away_new= pd.DataFrame(df_away.away_lu.tolist())
    df_away_new = pd.concat([df_away, df_away_new], axis=1)
    df_away_new = pd.melt(df_away_new, id_vars=['amargin', 'atime', 'away_lu', 'season'], value_vars=[0,1,2,3,4])
    df_away_new.columns = [u'margin', u'time', u'lu', u'season', u'variable', u'player_id']
    df_away_new.drop(['lu', 'variable'], inplace=True, axis=1)

    df_home = df_new[['hmargin', 'htime', 'home_lu', 'season']]
    df_home['home_lu'] = df_home.home_lu.apply(eval)
    df_home_new= pd.DataFrame(df_home.home_lu.tolist())
    df_home_new = pd.concat([df_home, df_home_new], axis=1)
    df_home_new = pd.melt(df_home_new, id_vars=['hmargin', 'htime', 'home_lu', 'season'], value_vars=[0,1,2,3,4])
    df_home_new.columns = [u'margin', u'time', u'lu', u'season', u'variable', u'player_id']
    df_home_new.drop(['lu', 'variable'], inplace=True, axis=1)

    agg_df = pd.concat([df_home_new, df_away_new])
    agg_df.dropna(inplace=True)
    agg_df['player_id'] = agg_df['player_id'].astype(int)

    output_df = agg_df.groupby(['player_id', 'season']).sum()
    output_df_count =agg_df.groupby(['player_id', 'season']).count()

    output_df = pd.concat([output_df, output_df_count['margin'].rename('count')], axis=1)

    output_df.to_sql('agg_matchups', conn, if_exists='replace', index=True)


if __name__ == '__main__':
    conn = connect_sql(config)

    # seasons = [str(i) for i in range(2008, 2016)]
    # run_seasons(seasons)

    create_aggregated_db(conn)