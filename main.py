from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <h1>Welcome to Smart Student Portal</h1>

    <p>This is my first Python Full Stack Project.</p>

    <button>Add Student</button>

    <button>View Students</button>
    """

@app.get("/student")
def student():
    