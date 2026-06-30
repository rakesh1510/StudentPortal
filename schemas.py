import datetime

from pydantic import BaseModel

class StudentCreate(BaseModel):
    name: str
    age: int
    email: str
    course: str

class User_Create(BaseModel):
    name: str
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str


