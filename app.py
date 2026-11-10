import os
import sqlite3
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
CORS(app)

DB_PATH = os.getenv("DB_PATH", "lily.db")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=API_KEY.strip()) if API_KEY else None

LILY_PERSONALITY = """
You are Lily, a personal AI assistant.
You are feminine, warm, friendly, intelligent and helpful.
Speak naturally and explain complicated things simply.
Be honest and never pretend to perform actions you cannot perform.
"""


def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def initialize_database():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                title TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)


initialize_database()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Lily AI Backend is running!",
        "assistant": "Lily"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "ai_configured": client is not None,
        "model": MODEL
    })


@app.route("/api/chat", methods=["POST"])
def chat():

    if client is None:
        return jsonify({
            "success": False,
            "error": "Gemini API key is not configured. Check your .env file."
        }), 503

    data = request.get_json(silent=True) or {}

    message = data.get("message", "")
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "success": False,
            "error": "Please enter a valid message."
        }), 400

    message = message.strip()

    try:
        with get_db() as db:

            previous = db.execute("""
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 20
            """, (session_id,)).fetchall()

            conversation = [
                {
                    "role": row["role"],
                    "parts": [{"text": row["content"]}]
                }
                for row in reversed(previous)
            ]

            conversation.append({
                "role": "user",
                "parts": [{"text": message}]
            })

            response = client.models.generate_content(
                model=MODEL,
                contents=conversation,
                config=types.GenerateContentConfig(
                    system_instruction=LILY_PERSONALITY,
                    temperature=0.7,
                    max_output_tokens=1500
                )
            )

            reply = response.text

            if not reply:
                raise ValueError(
                    "Gemini returned an empty response. "
                    "The response may have been blocked."
                )

            now = datetime.now(timezone.utc).isoformat()

            db.execute("""
                INSERT INTO messages
                (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, "user", message, now))

            db.execute("""
                INSERT INTO messages
                (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, "model", reply, now))

        return jsonify({
            "success": True,
            "session_id": session_id,
            "reply": reply
        })

    except Exception as error:

        app.logger.exception("Gemini chat request failed")

        return jsonify({
            "success": False,
            "error": f"{type(error).__name__}: {str(error)}"
        }), 500


@app.route("/api/history/<session_id>", methods=["GET"])
def history(session_id):

    with get_db() as db:
        rows = db.execute("""
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id
        """, (session_id,)).fetchall()

    return jsonify({
        "success": True,
        "messages": [dict(row) for row in rows]
    })


@app.route("/api/tasks", methods=["POST"])
def create_task():

    data = request.get_json(silent=True) or {}

    title = data.get("title", "")
    session_id = data.get("session_id", "")

    if not isinstance(title, str) or not isinstance(session_id, str):
        return jsonify({
            "success": False,
            "error": "Invalid task details."
        }), 400

    title = title.strip()
    session_id = session_id.strip()

    if not title or not session_id:
        return jsonify({
            "success": False,
            "error": "Task title and session_id are required."
        }), 400

    task_id = str(uuid.uuid4())

    with get_db() as db:
        db.execute("""
            INSERT INTO tasks
            (id, session_id, title, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            task_id,
            session_id,
            title,
            datetime.now(timezone.utc).isoformat()
        ))

    return jsonify({
        "success": True,
        "task_id": task_id,
        "title": title
    }), 201


@app.route("/api/tasks/<session_id>", methods=["GET"])
def get_tasks(session_id):

    with get_db() as db:
        rows = db.execute("""
            SELECT * FROM tasks
            WHERE session_id = ?
            ORDER BY created_at DESC
        """, (session_id,)).fetchall()

    return jsonify({
        "success": True,
        "tasks": [dict(row) for row in rows]
    })


@app.route("/api/tasks/<task_id>", methods=["PATCH"])
def update_task(task_id):

    data = request.get_json(silent=True) or {}

    with get_db() as db:
        result = db.execute("""
            UPDATE tasks
            SET completed = ?
            WHERE id = ?
        """, (int(bool(data.get("completed", False))), task_id))

        if result.rowcount == 0:
            return jsonify({
                "success": False,
                "error": "Task not found."
            }), 404

    return jsonify({
        "success": True,
        "message": "Task updated."
    })


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id):

    with get_db() as db:
        result = db.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task_id,)
        )

        if result.rowcount == 0:
            return jsonify({
                "success": False,
                "error": "Task not found."
            }), 404

    return jsonify({
        "success": True,
        "message": "Task deleted."
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )