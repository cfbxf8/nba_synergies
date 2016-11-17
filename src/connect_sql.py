import sqlalchemy
import ConfigParser


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