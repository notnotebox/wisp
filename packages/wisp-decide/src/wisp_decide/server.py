"""Small HTTP server (standard library only): /decide, /v1/systemone (Jev-style requests), /health."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .engine import Wisp
from .prompt import BASE_MODEL, BASE_REVISION, TASKS

MODEL_NAME = "wisp-v0.1.2-4b"
MAX_BODY = 20_000


def as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def decide_request(wisp: Wisp, body: dict) -> dict:
    """Native format: {task?, text, question, options}."""
    d = wisp.decide(body.get("text"), body.get("question"), body.get("options"), task=body.get("task"))
    return {"result": d.as_dict()}


def systemone_request(wisp: Wisp, body: dict) -> dict:
    """Jev/TypeSafe-style format: {state, questions: {name: {type, instructions, criteria}}}.

    `noul` (yes/no) and `choice` are supported; `score` is not (Wisp was not trained on rating scales).
    An optional top-level `task` ("moderation", "username", "report_validity") picks the built-in policy.
    """
    state, questions = body.get("state"), body.get("questions")
    if state is None or not isinstance(questions, dict) or not questions:
        raise ValueError("body needs 'state' and a non-empty 'questions' object")
    text, task = as_text(state), body.get("task")
    answers, tokens = {}, 0
    for name, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"question '{name}' must be an object")
        kind = q.get("type")
        instructions = q.get("instructions")
        if kind == "noul":
            question = as_text(instructions) if instructions else "Is this true?"
            d = wisp.decide(text, question, ["YES", "NO"], task=task)
            answers[name] = {"type": "noul", "noul": d.probabilities["YES"], "confidence": d.confidence}
        elif kind == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"choice '{name}' needs a 'criteria' object with at least 2 options")
            glosses = {k: as_text(v) for k, v in criteria.items() if v}
            question = as_text(instructions) if instructions else "Choose the best option."
            d = wisp.decide(text, question, list(criteria), task=task, glosses=glosses)
            answers[name] = {"type": "choice", "choice": d.decision, "probabilities": d.probabilities,
                             "confidence": d.confidence}
        else:
            raise ValueError(f"question '{name}': type '{kind}' is not supported (use 'noul' or 'choice')")
        tokens += getattr(wisp, "last_tokens", 0)
    return {"model": MODEL_NAME, "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0, "cached_input_tokens": 0}}


ROUTES = {"/decide": decide_request, "/v1/systemone": systemone_request}


def make_handler(wisp: Wisp):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code: int, body: dict) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            # Lets an https page (such as the benchmark site) call this local server in Chrome-based browsers.
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                return self.send_json(200, {"runtime": "wisp-decide", "status": "ok", "models": [{
                    "id": MODEL_NAME, "label": "Wisp v0.1.2 4B", "base_model": BASE_MODEL, "revision": BASE_REVISION,
                    "tasks": list(TASKS), "calibration": "uncalibrated", "automatic_punishment": False}]})
            if self.path != "/health":
                return self.send_json(404, {"error": "not_found"})
            self.send_json(200, {"status": "ok", "model": MODEL_NAME, "base": BASE_MODEL, "version": __version__,
                                 "calibration": "uncalibrated", "automatic_punishment": False})

        def do_POST(self) -> None:  # noqa: N802
            handler = ROUTES.get(self.path)
            if handler is None:
                return self.send_json(404, {"error": "not_found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= MAX_BODY:
                    raise ValueError(f"request size must be 1..{MAX_BODY} bytes")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                with lock:
                    result = handler(wisp, body)
                self.send_json(200, result)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(400 if self.path == "/decide" else 422, {"error": str(error)})
            except Exception as error:  # noqa: BLE001
                self.send_json(500, {"error": "inference_failed", "detail": str(error)})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def serve(wisp: Wisp, host: str = "127.0.0.1", port: int = 8781) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(wisp))
    print(json.dumps({"status": "listening", "host": host, "port": port}), flush=True)
    server.serve_forever()
