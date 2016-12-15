from pandas.io.json import json_normalize
import pandas as pd
import os
import json
import numpy as np
from helper_functions import connect_sql, read_one
from home_away_table import home_away_table_from_csv


class NBAParser():
    """Class to parse raw NBAstats json game files from a list of seasons.
    Intended purpose is for gathering data for computing Synergy Graph, but 
    creates a variety of tables that could have other uses.

    Original Dataset can be found here:
    http://statsnba.s3-website-us-east-1.amazonaws.com/
    Or gathered using: https://github.com/ethanluoyc/statsnba-playbyplay

    Parameters
    ----------
    season_list : list of strings ex. ['2014', '2015']
        Seasons to be looped through. 'ALL' = 2008-2015
    data_path : string
        Path to locally stored data.
    update_tables : list of strings
        SQL Tables to update.
        'ALL' = players, teams, matchups, team_stats, starter_stats, player_stats

    Attributes
    ----------
    players : pandas DataFrame
        DataFrame of all players
    teams : pandas DataFrame
        DataFrame of all teams
    matchups : pandas DataFrame
        DataFrame of all matchups for each game
    team_stats : pandas DataFrame
        DataFrame of all team stats for each game
    starter_stats : pandas DataFrame
        DataFrame of all team stats for each game
    player_stats : pandas DataFrame
        DataFrame of all player stats for each game
    """

    def __init__(self, season_list='ALL', data_path='../data/raw/matchups/',         update_tables='ALL'):
        self.data_path = data_path
        self.season_list = season_list
        self.update_tables = update_tables

        self.players = pd.DataFrame()
        self.teams = pd.DataFrame()
        self.matchups = pd.DataFrame()
        self.team_stats = pd.DataFrame()
        self.starter_stats = pd.DataFrame()
        self.player_stats = pd.DataFrame()

        self._con = None
        self._temp_season_obj = None

    def _connect_sql(self):
        """Connect to Local PostgresSQL DB"""
        self._con = connect_sql()

    def run_seasons(self):
        """Update SQL databases from locally stored NBA stats json files.

        Parameters
        ----------
        seasons : list of string ex. '2014'
            Seasons to be looped through
        """
        self._define_season_list()
        self._define_update_tables()

        for season in self.season_list:
            self._temp_season_obj = Season(season, data_path=self.data_path,
                                           update_tables=self.update_tables)
            self._add_season_dfs_to_master_dfs()

        self.update_sql()

    def update_sql(self):
        """Create or replace SQL databases depending on update_table list."""

        self._connect_sql()

        if 'players' in self.update_tables:
            self.players = self.players.drop_duplicates()
            self.players = self.players.reset_index(drop=True)
            self.players.to_sql('players_lookup', self._con,
                                if_exists='replace', index=False)

        if 'teams' in self.update_tables:
            self.teams = self.teams.drop_duplicates('id', keep='last')
            self.teams = self.teams.reset_index(drop=True)
            self.teams.to_sql('teams_lookup', self._con,
                              if_exists='replace', index=False)

        if 'matchups' in self.update_tables:
            self.matchups.to_sql('matchups', self._con,
                                 if_exists='replace', index=True)

        if 'team_stats' in self.update_tables:
            self.team_stats.to_sql('team_stats', self._con,
                                   if_exists='replace', index=False)

        if 'starter_stats' in self.update_tables:
            self.starter_stats.to_sql('starter_stats', self._con,
                                      if_exists='replace', index=False)

        if 'player_stats' in self.update_tables:
            self.player_stats.to_sql('player_stats', self._con,
                                     if_exists='replace', index=False)

        return "Finished"

    def _define_season_list(self):
        """Defines season_list based on initial user input."""
        if self.season_list == 'ALL':
            self.season_list = [str(i) for i in range(2008, 2016)]
        elif type(self.season_list) is str:
            self.season_list = [self.season_list]

    def _define_update_tables(self):
        """Defines update_tables list based on initial user input."""
        if self.update_tables == 'ALL':
            self.update_tables = ['players', 'teams', 'matchups', 'team_stats',
                                  'starter_stats', 'player_stats']
        elif type(self.update_tables) is str:
            self.update_tables = [self.update_tables]

    def _add_season_dfs_to_master_dfs(self):
        """Loop through each table and append seasons dfs to full master df with all seasons"""
        for i in self.update_tables:
            self._append_to_self(i)

    def _append_to_self(self, df_name_string):
        """Function that allows appending to table attributes in class.
        Left_df is the table attribute based on the df_name_string and
        right_df is the same attribute in the season object.
        Finally, set the table attribute as the combined df.

        Parameters
        ----------
        df_name_string : string
            table attribute in class to be appended
        """
        left_df = getattr(self, df_name_string)
        right_df = getattr(self._temp_season_obj, df_name_string)
        combined_df = pd.concat([left_df, right_df]).reset_index(drop=True)

        setattr(self, df_name_string, combined_df)


class Season():
    """Class to parse one season of raw NBAstats json files.
    Intended purpose is for gathering data for computing Synergy Graph, but 
    creates a variety of tables that could have other uses.

    Parameters
    ----------
    season : string ex. '2014'.
        Seasons to be looped through. 'ALL' = 2008-2015
    data_path : string
        Path to locally stored data.
    update_tables : list of strings
        SQL Tables to update.
        'ALL' = players, teams, matchups, team_stats, starter_stats, player_stats

    Attributes
    ----------
    players : pandas DataFrame
        DataFrame of all players
    teams : pandas DataFrame
        DataFrame of all teams
    matchups : pandas DataFrame
        DataFrame of all matchups for each game
    team_stats : pandas DataFrame
        DataFrame of all team stats for each game
    starter_stats : pandas DataFrame
        DataFrame of all team stats for each game
    player_stats : pandas DataFrame
        DataFrame of all player stats for each game
    """

    def __init__(self, season,  data_path='../data/raw/matchups/',
                 update_tables='ALL'):
        self.data_path = data_path
        self.season = season
        self.update_tables = update_tables

        self.players = None
        self.teams = None
        self.matchups = None
        self.team_stats = None
        self.starter_stats = None
        self.player_stats = None

        self._temp_game_obj = None
        self.get_season_data()

    def get_season_data(self):
        """Get game filenames based on data directory.
        Then run each these through Game class and append to full df."""
        file_path = self.data_path + self.season + \
                '-' + str(int(self.season[-2:])+1).zfill(2)

        for filename in os.listdir(file_path):
            gameid = filename.strip('matchups_').strip('.json')
            self._temp_game_obj = Game(gameid, self.data_path)
            self._update_dfs()

        for i in self.update_tables[2:]:
            self._add_season_column(i)

    def _update_dfs(self):
        """Update each df in update_tables by appending to full df."""
        if self.update_tables == 'ALL':
            self.update_tables = ['players', 'teams', 'matchups', 'team_stats',
                                  'starter_stats', 'player_stats']

        for i in self.update_tables:
            self._append_to_self(i)
            if (i == 'players') | (i == 'teams'):
                self._remove_duplicates(i)

    def _append_to_self(self, df_name_string):
        """Function that allows appending to table attributes in class.
        Left_df is the table attribute based on the df_name_string and
        right_df is the same attribute in the season object.
        Finally, set the table attribute as the combined df.

        Parameters
        ----------
        df_name_string : string
            table attribute in class to be appended
        """
        left_df = getattr(self, df_name_string)
        right_df = getattr(self._temp_game_obj, df_name_string)
        combined_df = pd.concat([left_df, right_df]).reset_index(drop=True)
        setattr(self, df_name_string, combined_df)

    def _add_season_column(self, df_name_string):
        """Function that adds season column to df.
        Allows for easier subsetting of table when querying SQL DB later.

        Parameters
        ----------
        df_name_string : string
            table attribute in class to add season column
        """
        df = getattr(self, df_name_string)
        df['season'] = self.season
        setattr(self, df_name_string, df)

    def _remove_duplicates(self, df_name_string):
        """Function to remove duplicates in df.

        Parameters
        ----------
        df_name_string : string
            table attribute in class to remove duplicates in
        """
        df = getattr(self, df_name_string)

        if (df_name_string == 'teams'):
            df = df.drop_duplicates('id', keep='last')
        else:
            df = df.drop_duplicates()
        setattr(self, df_name_string, df)


class Game():
    """Class to parse one game of raw NBAStats json files.
    Intended purpose is for gathering data for computing Synergy Graph, but 
    creates a variety of tables that could have other uses.

    Parameters
    ----------
    gameid : string ex. '0021500001'.
        gameid based on NBAStats.com game ids.
    data_path : string
        Path to locally stored data.

    Attributes
    ----------
    season : string
        season that game was played
    jso : json object
        loaded in json object from raw local json file
    players : pandas DataFrame
        DataFrame of all players
    teams : pandas DataFrame
        DataFrame of all teams
    matchups : pandas DataFrame
        DataFrame of all matchups for each game
    team_stats : pandas DataFrame
        DataFrame of all team stats for each game
    starter_stats : pandas DataFrame
        DataFrame of all team stats for each game
    player_stats : pandas DataFrame
        DataFrame of all player stats for each game
    """

    def __init__(self, gameid, data_path='../data/raw/matchups/'):
        self.gameid = gameid
        self.data_path = data_path
        self.season = None
        self.jso = None

        self.players = None
        self.teams = None
        self.matchups = None
        self.team_stats = None
        self.starter_stats = None
        self.player_stats = None

        self._home_df = None
        self._con = None
        self._is_home_correct = None
        self._get_game_data()

    def _get_game_data(self):
        """Run all necessary functions to get_game_data.
        Start with loading the JSON file and then checking for the correct 
        home_away ids. Then run through all 6 tables. """
        self._load_json()
        self._is_home_away_table_created()
        self._check_home_away()

        self._get_players()
        self._get_teams()
        self._get_matchups()
        self._get_team_stats()
        self._get_starter_stats()
        self._get_player_stats()

    def _load_json(self):
        """Load JSON file from gameid and datapath.
        Season and season directory can be parsed from gameid."""
        self.season = '20' + self.gameid[3:5]
        two_years = self.season + '-' + str(int(self.season[-2:])+1).zfill(2)
        file_path = self.data_path + two_years
        filename = 'matchups_' + self.gameid + '.json'
        with open(file_path + '/' + filename) as json_data:
            print filename
            self.jso = json.load(json_data)
        json_data.close()

    def _get_players(self):
        """Get players DataFrame from JSON."""
        home_players = json_normalize(self.jso['home_players'])[['id', 'name']]
        away_players = json_normalize(self.jso['away_players'])[['id', 'name']]

        self.players = pd.concat([home_players, away_players])

    def _get_teams(self):
        """Get teams DataFrame from JSON."""
        home_team = json_normalize(self.jso['home_team'])
        away_team = json_normalize(self.jso['away_team'])

        self.teams = pd.concat([home_team, away_team])

    def _get_matchups(self):
        """Get matchups DataFrame from JSON.
        This is the lineups, time, and point margins that occured at any
        substitution within the given game.
        Also checks to see if the home & away ids are correct and corrects."""
        if self.gameid != self.jso['game_id']:
            raise ValueError('Gameids do not match')

        home_lineup, away_lineup = [], []
        home_margin, away_margin = [], []
        home_time, away_time = [], []

        for i in self.jso['matchups']:
            home_lineup.append(tuple([x['id'] for x in i['home_players'][0]]))
            away_lineup.append(tuple([x['id'] for x in i['away_players'][0]]))

            home_diff = self._compute_pt_diff(i)

            home_margin.append(home_diff)
            away_margin.append(home_diff * -1)

            home_time.append(i['elapsed_seconds'])
            away_time.append(i['elapsed_seconds'])

        self._check_home_away()

        if self._is_home_correct:
            self.matchups = pd.DataFrame({"home_lineup": home_lineup,
                                          "away_lineup": away_lineup,
                                          "home_margin": home_margin,
                                          "away_margin": away_margin,
                                          "home_time": home_time,
                                          "away_time": away_time})

            self.matchups["GAME_ID"] = self.gameid
            self.matchups["home_id"] = int(self._home_df['home_ids'])
            self.matchups["away_id"] = int(self._home_df['away_ids'])
            self.matchups["was_home_correct"] = self._is_home_correct

        elif self._is_home_correct is False:
            self.matchups = pd.DataFrame({"home_lineup": away_lineup,
                                          "away_lineup": home_lineup,
                                          "home_margin": away_margin,
                                          "away_margin": home_margin,
                                          "home_time": away_time,
                                          "away_time": home_time})

            self.matchups["GAME_ID"] = self.gameid
            self.matchups["home_id"] = int(self._home_df['home_ids'])
            self.matchups["away_id"] = int(self._home_df['away_ids'])
            self.matchups["was_home_correct"] = self._is_home_correct

    def _get_team_stats(self):
        """Get team stats DataFrame from JSON."""
        self.team_stats = json_normalize(
            self.jso['_boxscore']['resultSets']['TeamStats'])

        mask = self.team_stats['TEAM_ID'] == int(self._home_df['home_ids'])
        self.team_stats['home_away'] = np.where(mask, 'home', 'away')

    def _get_starter_stats(self):
        """Get starter and bench stats DataFrame from JSON."""
        self.starter_stats = json_normalize(
            self.jso['_boxscore']['resultSets']['TeamStarterBenchStats'])

    def _get_player_stats(self):
        """Get player stats DataFrame from JSON."""
        self.player_stats = json_normalize(
            self.jso['_boxscore']['resultSets']['PlayerStats'])

    def _is_home_away_table_created(self):
        """Check to see if corrected home & away SQL DB is already created.
        If not, create it from csv file."""
        self._connect_sql()
        if self._con.dialect.has_table(self._con, "home_away") is False:
            home_away_table_from_csv()
        self._con.close()
        self._con.engine.dispose()

    def _check_home_away(self):
        """Get home and away ids from corrected SQL DB.
        Check to see if they are the same as from the json file.
        If not, make note of this to correct elsewhere."""
        self._home_df = read_one('home_away', 'GAME_ID', self.gameid)
        real_home_id = int(self._home_df['home_ids'])
        matchup_home_id = self.jso['home_team']['id']

        self._is_home_correct = matchup_home_id == real_home_id

    def _connect_sql(self):
        """Connect to SQL DB."""
        self._con = connect_sql()

    def _compute_pt_diff(self, one_matchup):
        """Compute point differential between lineups."""
        home_pts = one_matchup['home_end_score'] - one_matchup['home_start_score']
        away_pts = one_matchup['away_end_score'] - one_matchup['away_start_score']

        home_diff = home_pts - away_pts
        return home_diff


if __name__ == '__main__':
    # Get all seasons:
    # parser = NBAParser()

    # Get one season:
    # seas = Season('2015')

    # Get one game:
    game = Game('0021500001')
