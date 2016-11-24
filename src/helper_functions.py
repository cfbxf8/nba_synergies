import pandas as pd
import sqlalchemy
import ConfigParser
import time
import cPickle as pickle


def connect_sql():
    """Connect to SQL database using a config file and sqlalchemy
    """

    config = ConfigParser.ConfigParser()
    config.readfp(open('config.ini'))

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


def read_all(table):
    con = connect_sql()
    sql = 'SELECT * from ' + table + ';'
    df = pd.read_sql(sql=sql, con=con)

    if table == 'matchups':
        df['home_lineup'] = df.home_lineup.apply(eval)
        df['away_lineup'] = df.away_lineup.apply(eval)
    if table == 'matchups_reordered':
        df['i_lineup'] = df.i_lineup.apply(eval)
        df['j_lineup'] = df.j_lineup.apply(eval)

    return df


def read_season(table, season):
    con = connect_sql()
    condition = table + ".season = cast(" + season + " as text)"
    sql = 'SELECT * from ' + table + ' where ' + condition + ";"
    df = pd.read_sql(sql=sql, con=con)

    if table == 'matchups':
        df['home_lineup'] = df.home_lineup.apply(eval)
        df['away_lineup'] = df.away_lineup.apply(eval)
    if table == 'matchups_reordered':
        df['i_lineup'] = df.i_lineup.apply(eval)
        df['j_lineup'] = df.j_lineup.apply(eval)

    return df


def read_one(table, where_column, condition):
    con = connect_sql()

    if type(condition) is str:
        where = ' where "' + where_column + '"= '
        conditional = "'" + condition + "';"
        sql = 'SELECT * from ' + table + where + conditional
        df = pd.read_sql(sql=sql, con=con)

    if type(condition) is int:
        condition = where_column + "= " + condition + ";"
        sql = 'SELECT * from ' + table + ' where ' + condition

    if table == 'matchups':
        df['home_lineup'] = df.home_lineup.apply(eval)
        df['away_lineup'] = df.away_lineup.apply(eval)
    if table == 'matchups_reordered':
        df['i_lineup'] = df.i_lineup.apply(eval)
        df['j_lineup'] = df.j_lineup.apply(eval)

    return df


def subset_division(df, division):
    team_ids = read_all('teams_lookup')[['id', 'division']]
    df = df.merge(team_ids, how='left', left_on='i_id', right_on='id')
    df = df.merge(team_ids, how='left', left_on='j_id', right_on='id', suffixes=['_i', '_j'])
    df.drop(['id_i', 'id_j'], axis=1, inplace=True)

    df = df[(df['division_i'] == division) & (df['division_j'] == division)]

    df.reset_index(drop=True, inplace=True)
    return df


def before_date_df(df, last_day):
    df['date'] = pd.to_datetime(df.date, format="%Y-%m-%d")
    df = df[df['date'] <= last_day]
    return df


def read_pickle(file_name):
    with open(file_name, 'rb') as f:
        d = pickle.load(f)
    return d


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result

    return timed