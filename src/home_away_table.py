import ConfigParser
import pandas as pd
import requests
from pymongo import MongoClient
from connect_sql import connect_sql
from helper_functions import read_all

client = MongoClient()
db = client.nba
raw_table = db.raw

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))


def home_away_table(mongo_collection, psql_conn):
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

            mongo_collection.update_one(
                {"_id": i}, {"$set": result}, upsert=True)

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


def add_datetime():
    conn = connect_sql()
    old_df = read_all('home_away')
    db_client = MongoClient()
    db = db_client['nba']
    table = db['raw']

    jso = table.find()
    game_id, date = [], []
    for i in jso:
        date.append(i['rowSet'][0][0][:10])
        game_id.append(i['rowSet'][0][2])

    game_id = pd.DataFrame(game_id, columns=['game_id'])
    date = pd.DataFrame(date, columns=['date'])

    df = pd.concat([game_id, date], axis=1)
    df = df.rename(columns={"game_ids": "GAME_ID"})
    df.drop('game_id', axis=1, inplace=True)

    df = pd.concat([old_df, df], axis=1)

    df.to_sql('home_away', conn, if_exists='replace', index=False)

    return "Finished"


if __name__ == '__main__':
    conn = connect_sql()
    home_away_table(raw_table, conn)
