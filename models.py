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

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(100), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)

    file_size = Column(Integer)
    file_type = Column(String(50))

    uploaded_by = Column(Integer, nullable=False)

    total_pages = Column(Integer)
    total_chunks = Column(Integer)

    embedding_model = Column(String(100))
    vector_collection = Column(String(100))

    status = Column(String(30), default="processing")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), nullable=False, index=True)
    user_email = Column(String(100), nullable=False)
    title = Column(String(100), nullable=False)
    document_id = Column(String(100))   # <-- Add this
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), nullable=False, index=True)
    role = Column(String(100), nullable=False)
    content = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
