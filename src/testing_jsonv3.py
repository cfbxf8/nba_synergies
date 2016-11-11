import pandas as pd
import json
from collections import defaultdict
import os

def get_info(json_data):
	gameid = json_data['game_id']
	
	homeid = json_data['home_team']['id']
	awayid = json_data['away_team']['id']
	
	home_starters = tuple([x['id'] for x in json_data['home_starters']])
	away_starters = tuple([x['id'] for x in json_data['away_starters']])

	home_lu = []
	away_lu = []
	hmargin = []
	amargin = []
	htime = []
	atime = []
	# df = pd.DataFrame()
	for i in json_data['matchups']:
		home_lu.append(tuple([x['id'] for x in i['home_players'][0]]))
		away_lu.append(tuple([x['id'] for x in i['away_players'][0]]))

		hmargin.append(i['point_difference'] * -1)
		amargin.append(i['point_difference'])
		
		htime.append(i['elapsed_seconds'])
		atime.append(htime)
	
	temp_df = pd.DataFrame({"home_lu":home_lu, "away_lu":away_lu, 
		"hmargin":hmargin, "amargin":amargin, "htime":htime, 
		"atime":atime})

	# temp_df = temp_df.groupby(['home_lu',"away_lu"]).sum()
	
	temp_df["gameid"] = gameid
	temp_df["homeid"] = homeid
	temp_df["awayid"] = awayid

	# df = pd.concat([df, temp_df])

	return temp_df

if __name__ == '__main__':
	games = pd.DataFrame()
	for filename in os.listdir('/Users/clayton/Downloads/data/matchups/2008-09'):
		with open('/Users/clayton/Downloads/data/matchups/2008-09/' + filename) as json_data:
			d = json.load(json_data)
			print filename

		games = pd.concat([games, get_info(d)])
    