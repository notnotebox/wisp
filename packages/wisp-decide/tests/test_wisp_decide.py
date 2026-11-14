import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from wisp_decide import Wisp, build_prompt
from wisp_decide.prompt import CHAT_POLICY, SYMBOLS
from wisp_decide.server import make_handler


class FakeWisp(Wisp):
    """Prefers option index `pick`; records the prompts it was given."""

    def __init__(self, pick=0):
        super().__init__()
        self.pick, self.prompts, self.last_tokens = pick, [], 7

    def option_logits(self, prompt, n):
        self.prompts.append(prompt)
        return [4.0 if i == self.pick else 0.0 for i in range(n)]


class PromptTest(unittest.TestCase):
    def test_matches_reference_service_layout(self):
        got = build_prompt("hello", "Q?", ["REVIEW", "NO_ACTION"], "REPORTED MESSAGE", CHAT_POLICY)
        want = (f"CONTEXT\n{CHAT_POLICY}\n\nSTATE\nREPORTED MESSAGE\nhello\n\nQUESTION\nQ?\n\nOPTIONS\n"
                f"{SYMBOLS[0]}. REVIEW\n{SYMBOLS[1]}. NO_ACTION")
        self.assertEqual(got, want)

    def test_gloss_format(self):
        self.assertIn("A. THREAT - credible threat", build_prompt("x", "Q?", ["THREAT", "NONE"], glosses={"THREAT": "credible threat"}))


class DecideTest(unittest.TestCase):
    def test_decision_is_an_offered_option(self):
        d = FakeWisp(pick=1).decide("hi", "Q?", ["REVIEW", "NO_ACTION"])
        self.assertEqual(d.decision, "NO_ACTION")
        self.assertAlmostEqual(sum(d.probabilities.values()), 1.0)
        self.assertFalse(d.needs_review)

    def test_low_confidence_flags_review(self):
        w = FakeWisp()
        w.option_logits = lambda p, n: [0.1, 0.0]
        self.assertTrue(w.decide("hi", "Q?", ["A1", "B1"]).needs_review)

    def test_validation(self):
        w = FakeWisp()
        for bad in (("", "Q?", ["a", "b"]), ("t", "", ["a", "b"]), ("t", "Q?", ["a"]), ("t", "Q?", ["a", "a"]),
                    ("t", "Q?", [str(i) for i in range(13)])):
            with self.assertRaises(ValueError):
                w.decide(*bad)
        with self.assertRaises(ValueError):
            w.decide("t", "Q?", ["a", "b"], task="unknown_task")


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wisp = FakeWisp()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.wisp))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_health(self):
        with urllib.request.urlopen(self.base + "/health") as r:
            self.assertEqual(json.load(r)["automatic_punishment"], False)

    def test_models_and_cors(self):
        with urllib.request.urlopen(self.base + "/v1/models") as r:
            self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
            self.assertEqual(json.load(r)["models"][0]["id"], "wisp-v0.1.2-4b")

    def test_decide(self):
        code, out = self.post("/decide", {"task": "moderation", "text": "hi", "question": "Q?", "options": ["REVIEW", "NO_ACTION"]})
        self.assertEqual((code, out["result"]["decision"]), (200, "REVIEW"))
        self.assertIn("CHAT POLICY", self.wisp.prompts[-1])
        self.assertEqual(self.post("/decide", {"text": "hi"})[0], 400)

    def test_systemone_choice_and_noul(self):
        code, out = self.post("/v1/systemone", {
            "state": "The app crashes every time.",
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "technical": None}},
                "bug": {"type": "noul", "instructions": "Is this a bug report?"}}})
        self.assertEqual(code, 200)
        self.assertEqual(out["answers"]["team"]["choice"], "billing")
        self.assertIn("A. billing - payments", self.wisp.prompts[-2])
        self.assertTrue(0 < out["answers"]["bug"]["noul"] < 1)
        self.assertEqual(out["usage"]["output_tokens"], 0)

    def test_systemone_rejects_score_and_bad_input(self):
        code, out = self.post("/v1/systemone", {"state": "x", "questions": {"s": {"type": "score", "criteria": ["a", "b"]}}})
        self.assertEqual(code, 422)
        self.assertIn("not supported", out["error"])
        self.assertEqual(self.post("/v1/systemone", {"state": "x"})[0], 422)


if __name__ == "__main__":
    unittest.main()
