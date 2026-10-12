from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import OUTPUT_DIR
from .dataset_loader import Segment

REVIEW_DB_PATH = OUTPUT_DIR / 'editorial_reviews.db'

VALID_STATUSES = {"pendente", "aprovado", "aprovado_com_edicao", "rejeitado"}
RUBRIC_FIELDS = [
    "relevancia_jornalistica",
    "clareza",
    "fidelidade_factual",
    "adequacao_titulo",
    "adequacao_lead",
    "adequacao_corpo",
    "necessidade_correcao_humana",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class HumanReviewRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or REVIEW_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_batches (
                    batch_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    filename TEXT,
                    municipality TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_items (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    municipality TEXT,
                    minute_id TEXT,
                    item_title TEXT,
                    subject_text TEXT,
                    full_text TEXT,
                    iip_total REAL DEFAULT 0,
                    iip_level TEXT DEFAULT '',
                    iip_scores TEXT DEFAULT '{}',
                    binary_probability REAL,
                    binary_threshold REAL,
                    iip_direct REAL,
                    iip_derived REAL,
                    priority_rank INTEGER,
                    classification_model_version TEXT DEFAULT '',
                    generation_model_version TEXT DEFAULT '',
                    provenance_json TEXT DEFAULT '{}',
                    generation_strategy TEXT DEFAULT '',
                    generated_antetitle TEXT DEFAULT '',
                    generated_title TEXT DEFAULT '',
                    generated_superlead TEXT DEFAULT '',
                    generated_lead TEXT DEFAULT '',
                    generated_body TEXT DEFAULT '',
                    review_status TEXT DEFAULT 'pendente',
                    reviewer_name TEXT DEFAULT '',
                    reviewer_comment TEXT DEFAULT '',
                    manual_title TEXT DEFAULT '',
                    manual_antetitle TEXT DEFAULT '',
                    manual_superlead TEXT DEFAULT '',
                    manual_lead TEXT DEFAULT '',
                    manual_body TEXT DEFAULT '',
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(batch_id, segment_id),
                    FOREIGN KEY(batch_id) REFERENCES review_batches(batch_id)
                );

                CREATE TABLE IF NOT EXISTS review_rubrics (
                    item_id INTEGER PRIMARY KEY,
                    relevancia_jornalistica INTEGER,
                    clareza INTEGER,
                    fidelidade_factual INTEGER,
                    adequacao_titulo INTEGER,
                    adequacao_lead INTEGER,
                    adequacao_corpo INTEGER,
                    necessidade_correcao_humana INTEGER,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES review_items(item_id)
                );

                CREATE TABLE IF NOT EXISTS review_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES review_items(item_id)
                );
                """
            )
            try:
                conn.execute("ALTER TABLE review_items ADD COLUMN iip_scores TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass
            migrations = {
                "binary_probability": "REAL",
                "binary_threshold": "REAL",
                "iip_direct": "REAL",
                "iip_derived": "REAL",
                "priority_rank": "INTEGER",
                "classification_model_version": "TEXT DEFAULT ''",
                "generation_model_version": "TEXT DEFAULT ''",
                "provenance_json": "TEXT DEFAULT '{}'",
                "generated_antetitle": "TEXT DEFAULT ''",
                "generated_superlead": "TEXT DEFAULT ''",
                "manual_antetitle": "TEXT DEFAULT ''",
                "manual_superlead": "TEXT DEFAULT ''",
            }
            for column, definition in migrations.items():
                try:
                    conn.execute(f"ALTER TABLE review_items ADD COLUMN {column} {definition}")
                except sqlite3.OperationalError:
                    pass

    def bootstrap_batch(self, batch_id: str, run_id: str, segments: list[Segment], filename: str = "") -> dict[str, Any]:
        municipality = segments[0].municipality if segments else ""
        now = utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO review_batches (batch_id, run_id, filename, municipality, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (batch_id, run_id, filename, municipality, now, now),
            )
            inserted = 0
            for seg in segments:
                if not seg.generated_news.get("titulo"):
                    continue
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO review_items (
                        batch_id, run_id, segment_id, municipality, minute_id, item_title,
                        subject_text, full_text, iip_total, iip_level, iip_scores,
                        binary_probability, binary_threshold, iip_direct, iip_derived, priority_rank,
                        classification_model_version, generation_model_version, provenance_json,
                        generation_strategy, generated_antetitle, generated_title, generated_superlead,
                        generated_lead, generated_body,
                        updated_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        run_id,
                        seg.segment_id,
                        seg.municipality,
                        seg.minute_id,
                        seg.item_title,
                        seg.subject_text,
                        seg.full_text,
                        float(seg.iip_total or 0),
                        seg.iip_level or "",
                        json.dumps(getattr(seg, "iip_scores", {}) or {}, ensure_ascii=False),
                        getattr(seg, "binary_probability", None),
                        getattr(seg, "binary_threshold", None),
                        getattr(seg, "iip_direct", None),
                        getattr(seg, "iip_derived", None),
                        getattr(seg, "priority_rank", None),
                        getattr(seg, "model_version", ""),
                        getattr(seg, "generation_model_version", ""),
                        json.dumps(getattr(seg, "provenance", {}) or {}, ensure_ascii=False),
                        seg.generated_news.get("strategy", ""),
                        seg.generated_news.get("antetitulo", ""),
                        seg.generated_news.get("titulo", ""),
                        seg.generated_news.get("superlead", ""),
                        seg.generated_news.get("lead", ""),
                        seg.generated_news.get("corpo", ""),
                        now,
                        now,
                    ),
                )
                if cur.rowcount:
                    inserted += 1
                    item_id = conn.execute(
                        "SELECT item_id FROM review_items WHERE batch_id=? AND segment_id=?",
                        (batch_id, seg.segment_id),
                    ).fetchone()[0]
                    self._insert_history(conn, item_id, "bootstrap", {"segment_id": seg.segment_id, "run_id": run_id})
            conn.execute("UPDATE review_batches SET updated_at=? WHERE batch_id=?", (now, batch_id))
        return {"batch_id": batch_id, "inserted": inserted, "municipality": municipality}

    def list_items(self, batch_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT i.*, r.*
                FROM review_items i
                LEFT JOIN review_rubrics r ON r.item_id = i.item_id
                WHERE i.batch_id = ?
                ORDER BY i.item_id ASC
                """,
                (batch_id,),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_batch_summary(self, batch_id: str) -> dict[str, Any]:
        items = self.list_items(batch_id)
        counts = {s: 0 for s in VALID_STATUSES}
        for item in items:
            counts[item["review_status"]] = counts.get(item["review_status"], 0) + 1
        return {
            "batch_id": batch_id,
            "total": len(items),
            "counts": counts,
            "items_with_rubric": sum(1 for i in items if any(i["rubric"].values())),
        }

    def update_item(
        self,
        item_id: int,
        *,
        review_status: str | None = None,
        reviewer_name: str | None = None,
        reviewer_comment: str | None = None,
        manual_antetitle: str | None = None,
        manual_title: str | None = None,
        manual_superlead: str | None = None,
        manual_lead: str | None = None,
        manual_body: str | None = None,
        rubric: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        if review_status and review_status not in VALID_STATUSES:
            raise ValueError(f"Estado de revisão inválido: {review_status}")
        now = utcnow()
        payload = {}
        with self._connect() as conn:
            current = conn.execute("SELECT * FROM review_items WHERE item_id=?", (item_id,)).fetchone()
            if current is None:
                raise KeyError(f"Item não encontrado: {item_id}")
            fields = {}
            for key, value in {
                "review_status": review_status,
                "reviewer_name": reviewer_name,
                "reviewer_comment": reviewer_comment,
                "manual_antetitle": manual_antetitle,
                "manual_title": manual_title,
                "manual_superlead": manual_superlead,
                "manual_lead": manual_lead,
                "manual_body": manual_body,
            }.items():
                if value is not None:
                    fields[key] = value
                    payload[key] = value
            if fields:
                sets = ", ".join([f"{k}=?" for k in fields] + ["updated_at=?"])
                conn.execute(
                    f"UPDATE review_items SET {sets} WHERE item_id=?",
                    [*fields.values(), now, item_id],
                )
            if rubric is not None:
                clean = {k: (None if rubric.get(k) in (None, "") else int(rubric[k])) for k in RUBRIC_FIELDS}
                payload["rubric"] = clean
                conn.execute(
                    f"""
                    INSERT INTO review_rubrics (item_id, {', '.join(RUBRIC_FIELDS)}, updated_at)
                    VALUES (?, {', '.join(['?']*len(RUBRIC_FIELDS))}, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                    {', '.join([f'{f}=excluded.{f}' for f in RUBRIC_FIELDS])},
                    updated_at=excluded.updated_at
                    """,
                    [item_id, *[clean[f] for f in RUBRIC_FIELDS], now],
                )

            if payload:
                self._insert_history(conn, item_id, "update", payload)
        return self.get_item(item_id)

    def get_item(self, item_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT i.*, r.*
                FROM review_items i
                LEFT JOIN review_rubrics r ON r.item_id = i.item_id
                WHERE i.item_id=?
                """,
                (item_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Item não encontrado: {item_id}")
            history = conn.execute(
                "SELECT action, payload, created_at FROM review_history WHERE item_id=? ORDER BY history_id ASC",
                (item_id,),
            ).fetchall()
        item = self._row_to_item(row)
        item["history"] = [
            {"action": h["action"], "payload": json.loads(h["payload"]), "created_at": h["created_at"]}
            for h in history
        ]
        return item

    def export_review_dataset(self, batch_id: str, out_path: str | Path) -> Path:
        items = self.list_items(batch_id)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "batch_id": batch_id,
            "exported_at": utcnow(),
            "summary": self.get_batch_summary(batch_id),
            "items": items,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def _insert_history(self, conn: sqlite3.Connection, item_id: int, action: str, payload: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO review_history (item_id, action, payload, created_at) VALUES (?, ?, ?, ?)",
            (item_id, action, json.dumps(payload, ensure_ascii=False), utcnow()),
        )

    def _row_to_item(self, row: sqlite3.Row) -> dict[str, Any]:
        rubric = {f: row[f] for f in RUBRIC_FIELDS}
        generated = {
            "antetitulo": row["generated_antetitle"],
            "title": row["generated_title"],
            "superlead": row["generated_superlead"],
            "lead": row["generated_lead"],
            "body": row["generated_body"],
        }
        corrected = {
            "antetitulo": row["manual_antetitle"] or row["generated_antetitle"],
            "title": row["manual_title"] or row["generated_title"],
            "superlead": row["manual_superlead"] or row["generated_superlead"],
            "lead": row["manual_lead"] or row["generated_lead"],
            "body": row["manual_body"] or row["generated_body"],
        }
        return {
            "item_id": row["item_id"],
            "batch_id": row["batch_id"],
            "run_id": row["run_id"],
            "segment_id": row["segment_id"],
            "municipality": row["municipality"],
            "minute_id": row["minute_id"],
            "item_title": row["item_title"],
            "subject_text": row["subject_text"],
            "full_text": row["full_text"],
            "iip_total": row["iip_total"],
            "iip_level": row["iip_level"],
            "iip_scores": json.loads(row["iip_scores"] or "{}") if "iip_scores" in row.keys() else {},
            "binary_probability": row["binary_probability"],
            "binary_threshold": row["binary_threshold"],
            "iip_direct": row["iip_direct"],
            "iip_derived": row["iip_derived"],
            "priority_rank": row["priority_rank"],
            "classification_model_version": row["classification_model_version"],
            "generation_model_version": row["generation_model_version"],
            "provenance": json.loads(row["provenance_json"] or "{}"),
            "generation_strategy": row["generation_strategy"],
            "review_status": row["review_status"],
            "reviewer_name": row["reviewer_name"],
            "reviewer_comment": row["reviewer_comment"],
            "generated": generated,
            "corrected": corrected,
            "rubric": rubric,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
