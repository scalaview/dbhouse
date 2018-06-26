from datetime import datetime
from sqlalchemy import Column, String, Integer, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import mysql

import config
from sqlalchemy import create_engine
engine = create_engine(config.config["development"].SQLALCHEMY_DATABASE_URI, max_overflow=5)

Base = declarative_base()

class BaseModel(object):
    id = Column(Integer, primary_key=True)
    createdAt = Column(mysql.DATETIME(), nullable=False, default=datetime.utcnow)
    updatedAt = Column(mysql.DATETIME(), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

class DailyPrice(BaseModel, Base):
    __tablename__ = 'daily_prices'
    market = Column(String(64), nullable=False)
    fsymbol = Column(String(64), nullable=False)
    tsymbol = Column(String(64), nullable=False)
    date   = Column(mysql.DATETIME(), nullable=False)
    open_price  = Column(Numeric(precision=8, scale=2, asdecimal=False, decimal_return_scale=None), default=0.00)
    high_price  = Column(Numeric(precision=8, scale=2, asdecimal=False, decimal_return_scale=None), default=0.00)
    low_price  = Column(Numeric(precision=8, scale=2, asdecimal=False, decimal_return_scale=None), default=0.00)
    close_price  = Column(Numeric(precision=8, scale=2, asdecimal=False, decimal_return_scale=None), default=0.00)
    volumefrom  = Column(Numeric(precision=16, scale=2, asdecimal=False, decimal_return_scale=None), default=0.00)
    volumeto  = Column(Numeric(precision=16, scale=2, asdecimal=False, decimal_return_scale=None), default=0.00)
    evm_7 = Column(Numeric(precision=16, scale=8, asdecimal=False, decimal_return_scale=None), default=0.00)
    evm_14 = Column(Numeric(precision=16, scale=8, asdecimal=False, decimal_return_scale=None), default=0.00)
    cci_30 = Column(Numeric(precision=16, scale=8, asdecimal=False, decimal_return_scale=None), default=0.00)
    roc_30 = Column(Numeric(precision=16, scale=8, asdecimal=False, decimal_return_scale=None), default=0.00)
    forceIndex_1 = Column(Numeric(precision=32, scale=8, asdecimal=False, decimal_return_scale=None), default=0.00)
    tag = Column(Integer)
    # ALTER TABLE daily_prices ADD tag int(11);
