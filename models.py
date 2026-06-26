from sqlalchemy import column, Column
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import DateTime

from sqlalchemy.sql import func

from database import Base, engine

class Student(Base):
    __tablename__ = "student"
    id = Column(Integer, primary_key=True,index=True,
        autoincrement=True)
    name = Column(String)
    email = Column(String)
    age = Column(Integer)
    course = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True,index=True,autoincrement=True)
    name = Column(String)
    email = Column(String,unique=True)
    password = Column(String,nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())



