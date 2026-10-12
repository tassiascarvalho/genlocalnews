"""Aplicação GenLocalNews."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for
from flask_cors import CORS

load_dotenv()

from api.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from backend.routes import api_bp

# Contas de demonstração.
DEMO_USERS = {
    "inesctec": "inesctec",
    "ubi": "ubi",
}


def _safe_next(value: str | None) -> str:
    """Aceita apenas destinos locais para evitar redirecionamentos externos."""
    if not value:
        return "/"
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or not value.startswith("/"):
        return "/"
    return value


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.secret_key = os.getenv("GENLOCALNEWS_SECRET_KEY", "genlocalnews-demo-change-me")
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    CORS(app)

    app.register_blueprint(api_bp, url_prefix="/api")
    frontend_dir = Path(__file__).parent / "frontend"

    @app.before_request
    def require_login():
        if request.endpoint in {"login", "logout"}:
            return None
        if session.get("authenticated"):
            return None
        if request.path.startswith("/api/"):
            return jsonify({"error": "Autenticação necessária."}), 401
        return redirect(url_for("login", next=request.full_path if request.query_string else request.path))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = ""
        next_url = _safe_next(request.values.get("next"))
        if request.method == "POST":
            username = (request.form.get("username") or "").strip().lower()
            password = request.form.get("password") or ""
            expected = DEMO_USERS.get(username)
            if expected is not None and hmac.compare_digest(password, expected):
                session.clear()
                session["authenticated"] = True
                session["username"] = username
                return redirect(next_url)
            error = "Utilizador ou palavra-passe inválidos."
        return send_from_directory(str(frontend_dir), "login.html") if not error else _render_login_with_error(frontend_dir, error, next_url)

    def _render_login_with_error(frontend_dir: Path, error: str, next_url: str):
        template = (frontend_dir / "login.html").read_text(encoding="utf-8")
        template = template.replace("<!--LOGIN_ERROR-->", f'<div class="login-error">{error}</div>')
        template = template.replace('name="next" value="/"', f'name="next" value="{next_url}"')
        return template, 401, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    def index():
        return send_from_directory(str(frontend_dir), "integrated_pipeline.html")

    @app.route("/revisao")
    @app.route("/revisao.html")
    def review_page():
        return send_from_directory(str(frontend_dir), "revisao.html")

    # URLs antigas de resultados.
    @app.route("/citilink_editorial_review.html")
    def legacy_review_page():
        target = "/revisao.html"
        if request.query_string:
            target += "?" + request.query_string.decode("utf-8", errors="ignore")
        return redirect(target)

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(str(frontend_dir), filename)

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"GenLocalNews disponível em http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
