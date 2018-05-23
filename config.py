import os
import json
basedir = os.path.abspath(os.path.dirname(__file__))
config_path = basedir + "/config.json"
database_config = json.loads(open(config_path).read())

class Option():
    def __init__(self, d):
        self.__dict__ = d

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    SSL_DISABLE = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_RECORD_QUERIES = True
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
    FLASKY_MAIL_SENDER = 'Flask Admin <flask@example.com>'
    FLASKY_ADMIN = os.environ.get('FLASK_ADMIN')
    FLASKY_POSTS_PER_PAGE = 20
    FLASKY_FOLLOWERS_PER_PAGE = 50
    FLASKY_COMMENTS_PER_PAGE = 30
    FLASKY_SLOW_DB_QUERY_TIME=0.5

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    option = Option(database_config.get("development")["db"])
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'mysql://%s:%s@%s/%s' % (option.username, option.password, option.hostname, option.database)

class StagingConfig(Config):
    TESTING = True
    option = Option(database_config.get("staging")["db"])
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        'mysql://%s:%s@%s/%s' % (option.username, option.password, option.hostname, option.database)

class ProductionConfig(Config):
    option = Option(database_config.get("production")["db"])
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql://%s:%s@%s/%s' % (option.username, option.password, option.hostname, option.database)

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)


config = {
    'development': DevelopmentConfig,
    'staging': StagingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
