from _pyrepl import reader
from email.mime import text
from urllib import response

from click import prompt
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from langchain_core import documents
from langchain_core.embeddings import embeddings
from langsmith import client
from numpy.ma.core import ids
from sqlalchemy.orm import Session
from sqlalchemy.orm.collections import collection
from starlette.responses import FileResponse
from torch.distributed.autograd import context

from database import engine, SessionLocal
from models import Base, Student, User
from schemas import StudentCreate, User_Create, Login
from auth import hash_password, verify_password
from security import create_access_token, verify_token
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from pypdf import PdfReader
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat
import uuid

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


app = FastAPI()


@app.post("/students")
async def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    new_student = Student(name=student.name, age=student.age, email=student.email, course=student.course)

    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    return new_student


@app.get("/students")
async def get_students(db: Session = Depends(get_db), current_user: str = Depends(verify_token)):
    students = db.query(Student).all()
    return {
        "logged_in_user": current_user,
        "students": db.query(Student).all()
    }


@app.get("/students/{student_id}")
def Search_student(student_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == student_id).first()
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    else:
        return student


@app.put("/students/{student_id}")
def Update_Student(student_id: int, student: StudentCreate, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == student_id).first()
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    else:
        student.name = student.name
        student.age = student.age
        student.email = student.email
        student.course = student.course
        db.commit()
        db.refresh(student)
        return student


@app.delete("/students/{student_id}")
def Delete_Student(student_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == student_id).first()
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    else:
        db.delete(student)
        db.commit()
        db.refresh(student)
        return student


# @app.register("/reister/{student_username}/{student_pwd}")
@app.post("/register")
def Register_User(user: User_Create, db: Session = Depends(get_db)):
    print(user)
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    new_student = User(name=user.name, email=user.email, password=hash_password(user.password))
    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    return new_student


@app.post("/login")
async def login_user(login: Login, db: Session = Depends(get_db)):
    student = db.query(User).filter(User.email == login.email).first()
    if student:
        print("login.password =", login.password)
        print("student.password =", student.password)
        if verify_password(login.password, student.password):
            token = create_access_token({
                'sub': student.email,
            })
            return {
                "access_token": token,
                "token_type": "bearer"
            }
        else:
            raise HTTPException(status_code=400, detail="Incorrect email or password")


UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {
        'message': 'file uploaded successfully',
        'filename': file.filename,
        'path': file_path
    }


@app.post("/read-pdf")
async def read_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)

    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return {
        "content": text[:5000]
    }


# @app.post("/pdf-chunks")
# async def pdf_chunks(file: UploadFile = File(...)):
#     reader = PdfReader(file.file)
#     text = ''
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"
#
#         text_splitter=RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = text_splitter.split_text(text)
#         return {
#             "total_pages": len(reader.pages),
#             "total_text_length": len(text),
#             "text_chunks": len(chunks),
#             "first_chunk": chunks[0] if chunks else "No text found"
#         }
#

@app.post("/pdf-chunks")
async def pdf_chunks(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_debug = []
    text = ""

    for page_no, page in enumerate(doc):
        page_text = page.get_text("text")

        page_debug.append({
            "page": page_no + 1,
            "text_length": len(page_text),
            "preview": page_text[:100]
        })

        if page_text:
            text += page_text + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return {
        "total_pages": len(doc),
        "total_text_length": len(text),
        "text_chunks": len(chunks),
        "first_10_pages_debug": page_debug[:10],
        "first_chunk": chunks[0] if chunks else "No text found"
    }


@app.post("/pdf-store")
async def pdf_store(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ''
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text += page_text + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks).tolist()

    client = chromadb.PersistentClient(path='./chroma.db')
    collection = client.get_or_create_collection(name='pdf_document')

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    return {
        'message': "PDF store in ChromaDB successfully",
        'total chunks': len(chunks)
    }


@app.get('/search-pdf')
async def search_pdf(query: str, db: Session = Depends(get_db)):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embeddings = model.encode(query)
    client = chromadb.PersistentClient(path='./chroma.db')
    collection = client.get_or_create_collection(name='pdf_document')

    results = collection.query(query_embeddings=[query_embeddings],
                               n_results=1
                               )
    return {
        'question': query,
        'matched_chunk': results["documents"][0]
    }


@app.get("/ask-pdf")
async def ask_pdf(question: str, db: Session = Depends(get_db)):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode([question]).tolist()[0]

    client = chromadb.PersistentClient(path='./chroma.db')
    collection = client.get_or_create_collection(name='pdf_document')

    results = collection.query(
        query_embeddings=[question_embeddings],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
    You are a PDF question-answering assistant.

    Answer ONLY from the context below.
    The context is available to you.
    Do not say you do not have access to the document if context is provided.

    Context:
    {context}

    Question:
    {question}

    Give a short direct answer.
    """
    response = chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return {
        "question": question,
        "answer": response["message"]["content"],
        "source_chunks": results["documents"][0]
    }


#  PAHSE 2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222


@app.post('/pdf-store')
async def pdf_store(file: UploadFile = File(...)):
    document_id = str(uuid.uuid4())
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ''

    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text += page_text + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks).tolist()
    client = chromadb.PersistentClient(path='./chroma.db')
    collection = client.get_or_create_collection(name='pdf_document')

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{
        "document_id": document_id,
        "filename": file.filename,
        "chunk_id": i
    }]

    collection.adds(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
