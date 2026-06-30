from _pyrepl import reader
from email.mime import text
from urllib import response

from click import prompt
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from langchain_core import documents
from langchain_core.embeddings import embeddings
from langgraph_sdk._sync import assistants
from langsmith import client
from langsmith._openapi_client.resources.sessions import sessions
from numpy.ma.core import ids
from sqlalchemy.orm import Session
from sqlalchemy.orm.collections import collection
from sqlalchemy.sql import roles
from starlette.responses import FileResponse
from sympy.printing.cxx import reserved
from torch.distributed.autograd import context

from database import engine, SessionLocal
print("Database URL:", engine.url)
from models import Base, Student, User, Document, Conversation, Message
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
from fastapi.security import OAuth2PasswordRequestForm
# import init_db
import shutil

Base.metadata.create_all(bind=engine)
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("PRAGMA table_info(conversations);"))
    print("\n=== Conversations Table ===")
    for row in result:
        print(row)

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


# @app.post("/login")
# async def login_user(login: Login, db: Session = Depends(get_db)):
#     student = db.query(User).filter(User.email == login.email).first()
#     if student:
#         print("login.password =", login.password)
#         print("student.password =", student.password)
#         if verify_password(login.password, student.password):
#             token = create_access_token({
#                 'sub': student.email,
#             })
#             return {
#                 "access_token": token,
#                 "token_type": "bearer"
#             }
#         else:
#             raise HTTPException(status_code=400, detail="Incorrect email or password")

@app.post("/login")
async def login_user(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.email == form_data.username
    ).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token({
        "sub": user.email
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }


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


@app.post("/pdf-store-v1")
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

@app.get("/documents")
async def get_documents(
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    documents = db.query(Document).filter(
        Document.uploaded_by == 1      # Replace with your user logic later
    ).all()

    return documents

from models import Document
import uuid
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile, File, Depends
from sqlalchemy.orm import Session


@app.post("/pdf-store-v2")
async def pdf_store(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        current_user: str = Depends(verify_token)
):
    # Generate unique document id
    document_id = str(uuid.uuid4())

    # Save PDF locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    pdf_bytes = await file.read()

    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    # Open PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in doc:
        page_text = page.get_text("text")

        if page_text:
            text += page_text + "\n"

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    # Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    # Store document information in SQL
    new_document = Document(
        document_id=document_id,
        filename=file.filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=len(pdf_bytes),
        file_type=file.content_type,
        uploaded_by=1,
        total_pages=len(doc),
        total_chunks=len(chunks),
        embedding_model="all-MiniLM-L6-v2",
        vector_collection="pdf_document",
        status="processed"
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    # Store chunks in ChromaDB
    client = chromadb.PersistentClient(path="./chroma.db")

    collection = client.get_or_create_collection(
        name="pdf_document"
    )

    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "filename": file.filename,
            "chunk_index": i,
            "uploaded_by": current_user
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "message": "PDF stored successfully",
        "document_id": document_id,
        "filename": file.filename,
        "total_pages": len(doc),
        "total_chunks": len(chunks)
    }

@app.post("/pdf-store-v3")
async def pdf_store_v3(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    document_id = str(uuid.uuid4())

    pdf_bytes = await file.read()

    file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_texts = []

    for page_no, page in enumerate(doc):
        page_text = page.get_text("text")

        if page_text and page_text.strip():
            page_texts.append({
                "page": page_no + 1,
                "text": page_text
            })

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    metadatas = []

    for page_data in page_texts:
        page_chunks = text_splitter.split_text(page_data["text"])

        for index, chunk in enumerate(page_chunks):
            chunks.append(chunk)

            metadatas.append({
                "document_id": document_id,
                "filename": file.filename,
                "page_number": page_data["page"],
                "chunk_index": len(chunks) - 1,
                "uploaded_by": current_user
            })

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    new_document = Document(
        document_id=document_id,
        filename=file.filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=len(pdf_bytes),
        file_type=file.content_type,
        uploaded_by=1,
        total_pages=len(doc),
        total_chunks=len(chunks),
        embedding_model="all-MiniLM-L6-v2",
        vector_collection="pdf_document",
        status="processed"
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    client = chromadb.PersistentClient(path="./chroma.db")
    collection = client.get_or_create_collection(name="pdf_document")

    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "message": "PDF stored successfully with improved chunks",
        "document_id": document_id,
        "filename": file.filename,
        "total_pages": len(doc),
        "total_chunks": len(chunks)
    }


@app.get("/ask-pdf-global")
async def ask_pdf_global(
        question: str,
        current_user: str = Depends(verify_token)
):
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert user's question into a vector embedding
    question_embedding = model.encode([question]).tolist()[0]

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path="./chroma.db")

    # Open (or create) the PDF collection
    collection = client.get_or_create_collection(
        name="pdf_document"
    )

    # Search across ALL uploaded PDFs
    # Returns the 5 most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    # Extract matched text chunks
    chunks = results["documents"][0]

    # Extract metadata (filename, document_id, chunk number, etc.)
    metadatas = results["metadatas"][0]

    # Merge all retrieved chunks into one context
    context = "\n\n".join(chunks)

    # Prompt sent to the LLM
    prompt = f"""
You are a PDF question-answering assistant.

Answer ONLY from the context below.

If the answer is not available in the context, say:
"I could not find this information in the uploaded PDFs."

Context:
{context}

Question:
{question}

Give a short direct answer.
"""

    # Send prompt to local Ollama model
    response = chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    metadatas = results.get("metadatas", [[]])[0] or []

    sources = []

    for metadata in metadatas:
        metadata = metadata or {}

        sources.append({
            "document_id": metadata.get("document_id"),
            "filename": metadata.get("filename"),
            "chunk_index": metadata.get("chunk_index"),
            "uploaded_by": metadata.get("uploaded_by")
        })

    # Return the final answer along with source information
    return {
        "question": question,
        "answer": response["message"]["content"],
        "sources": sources,
        "source_chunks": chunks
    }


@app.post("/conversations")
async def conversations(
    document_id: str,
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    conversation_id = str(uuid.uuid4())

    new_conversation = Conversation(
        conversation_id=conversation_id,
        user_email=current_user,
        title="New Chat",
        document_id=document_id
    )

    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)

    return {
        "message": "conversation created",
        "conversation_id": conversation_id,
        "document_id": document_id
    }


@app.get("/ask-pdf-memory")
async def ask_pdf_memory(
    conversation_id: str,
    question: str,
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    import re

    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id,
        Conversation.user_email == current_user
    ).first()

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    document_id = conversation.document_id

    if not document_id:
        raise HTTPException(status_code=400, detail="Conversation has no document linked")

    # Load old messages BEFORE saving current question
    old_messages = db.query(Message).filter(
        Message.conversation_id == conversation.conversation_id
    ).order_by(Message.id.desc()).limit(10).all()

    old_messages = old_messages[::-1]

    original_question = question
    question_lower = question.lower()

    # Follow-up question support: next / previous command
    if "next command" in question_lower or "previous command" in question_lower:
        previous_command = None

        for msg in reversed(old_messages):
            match = re.search(r"\b(\d{5})\b", msg.content)
            if match:
                previous_command = int(match.group(1))
                break

        if previous_command:
            if "next command" in question_lower:
                question = f"What is command {previous_command + 1}?"
            elif "previous command" in question_lower:
                question = f"What is command {previous_command - 1}?"

    # Now save current user message
    user_message = Message(
        conversation_id=conversation.conversation_id,
        role="user",
        content=original_question
    )
    db.add(user_message)
    db.commit()

    chat_history = ""
    for msg in old_messages:
        chat_history += f"{msg.role}: {msg.content}\n"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode([question]).tolist()[0]

    client = chromadb.PersistentClient(path="./chroma.db")
    collection = client.get_or_create_collection(name="pdf_document")

    where_filter = {
        "$and": [
            {"document_id": document_id},
            {"uploaded_by": current_user}
        ]
    }

    all_results = collection.get(where=where_filter)
    all_chunks = all_results.get("documents", []) or []
    all_metadatas = all_results.get("metadatas", []) or []

    exact_chunks = []
    exact_metadatas = []

    command_match = re.search(r"\b\d{5}\b", question)

    if command_match:
        command_number = command_match.group(0)

        for chunk, metadata in zip(all_chunks, all_metadatas):
            if command_number in chunk:
                exact_chunks.append(chunk)
                exact_metadatas.append(metadata)

    semantic_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5,
        where=where_filter
    )

    semantic_chunks = semantic_results.get("documents", [[]])[0] or []
    semantic_metadatas = semantic_results.get("metadatas", [[]])[0] or []

    # Important: if exact command found, use only exact chunks
    if exact_chunks:
        chunks = exact_chunks[:5]
        metadatas = exact_metadatas[:5]
    else:
        chunks = semantic_chunks[:5]
        metadatas = semantic_metadatas[:5]

    if not chunks:
        answer = "I could not find this information in the selected PDF."
    else:
        context = "\n\n".join(chunks)

        prompt = f"""
You are an AI assistant answering questions from a PDF.

Rules:
- Answer ONLY from the PDF context.
- Use conversation history only to understand follow-up questions.
- Never invent information.
- If a command number is asked, answer only from chunks containing that exact command number.
- Do not answer from a similar number.
- If the answer is not found in the PDF context, reply:
  "I could not find this information in the selected PDF."
- Keep the answer clear and concise.

Conversation History:
{chat_history}

PDF Context:
{context}

Question:
{question}

Answer:
"""

        response = chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

    assistant_message = Message(
        conversation_id=conversation.conversation_id,
        role="assistant",
        content=answer
    )
    db.add(assistant_message)
    db.commit()

    sources = []

    for metadata in metadatas:
        metadata = metadata or {}
        sources.append({
            "document_id": metadata.get("document_id"),
            "filename": metadata.get("filename"),
            "page_number": metadata.get("page_number"),
            "chunk_index": metadata.get("chunk_index"),
            "uploaded_by": metadata.get("uploaded_by")
        })

    return {
        "conversation_id": conversation.conversation_id,
        "document_id": document_id,
        "original_question": original_question,
        "resolved_question": question,
        "answer": answer,
        "sources": sources
    }

@app.delete("/reset-pdf-data")
async def reset_pdf_data(
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # delete chat messages
    conversations = db.query(Conversation).filter(
        Conversation.user_email == current_user
    ).all()

    conversation_ids = [c.conversation_id for c in conversations]

    if conversation_ids:
        db.query(Message).filter(
            Message.conversation_id.in_(conversation_ids)
        ).delete(synchronize_session=False)

    # delete conversations
    db.query(Conversation).filter(
        Conversation.user_email == current_user
    ).delete(synchronize_session=False)

    # delete document records
    db.query(Document).filter(
        Document.uploaded_by == 1
    ).delete(synchronize_session=False)

    db.commit()

    # delete ChromaDB collection
    client = chromadb.PersistentClient(path="./chroma.db")

    try:
        client.delete_collection(name="pdf_document")
    except Exception:
        pass

    client.get_or_create_collection(name="pdf_document")

    return {
        "message": "Old PDF data deleted successfully"
    }



@app.delete("/reset-chroma-complete")
async def reset_chroma_complete():
    chroma_path = "./chroma.db"

    try:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        client = chromadb.PersistentClient(path=chroma_path)
        client.get_or_create_collection(name="pdf_document")

        return {
            "message": "ChromaDB completely reset",
            "collection": "pdf_document",
            "vectors": 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/check-chroma")
async def check_chroma():
        client = chromadb.PersistentClient(path="./chroma.db")
        collection = client.get_or_create_collection(name="pdf_document")

        return {
            "collection": "pdf_document",
            "total_vectors": collection.count()
        }