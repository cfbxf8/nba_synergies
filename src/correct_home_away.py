import pandas as pd
import numpy as np
from helper_functions import read_all, connect_sql


### TEST #####
team_stats = read_all('team_stats')
df = read_all('matchups')

home = team_stats[team_stats['home_away'] == 'home']
home_ = home[['GAME_ID', 'home_away', 'TEAM_ID', 'TEAM_ABBREVIATION' , 'PTS']]

away = team_stats[team_stats['home_away'] == 'away']
away_ = away[['GAME_ID', 'home_away', 'TEAM_ID', 'TEAM_ABBREVIATION' , 'PTS']]

total = home_.merge(away_, on='GAME_ID', suffixes=['_home', '_away'])

sum_ = df.groupby('GAME_ID').sum()[['home_margin', 'away_margin']]

first_ = df.groupby('GAME_ID').first()[['home_id', 'away_id', 'home_lineup', 'away_lineup', 'was_home_correct', 'season']]

sum_.reset_index(inplace=True)

all = total.merge(sum_, on='GAME_ID')

first_.reset_index(inplace=True)
all = all.merge(first_, on='GAME_ID')

all.to_csv('test_home_all2.csv')


#### CORRECT WHERE was_home_correct is TRUE ######
# df_new = df.copy()
# df_new['home_margin'] = np.where(df['was_home_correct'], df_new['home_margin'] * -1, df_new['home_margin'])

# df_new['away_margin'] = np.where(df['was_home_correct'], df_new['away_margin'] * -1, df_new['away_margin'])

# con = connect_sql()

# df_new.to_sql('matchups', con, if_exists='replace', index=True)