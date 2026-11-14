# wisp-decide

`wisp-decide` is the Python reference package for the experimental Wisp v0.1.2 release. It runs the trained adapter and provides an alternative to the native Rust server.

Small client and server for **Wisp v0.1.2 4B** (adapter: `notebox/Wisp-v0.1.2-4B`, base: `Qwen/Qwen3.5-4B`).
This package is the **secondary Python option**. For the primary Wisp runtime, use the native Rust/Candle server in the [Wisp repository](https://github.com/notnotebox/wisp) and follow its [runtime setup guide](https://github.com/notnotebox/wisp/blob/main/docs/RUST_RUNTIME.md).
It builds the prompt the adapter was trained with, reads the answer from the logits of your options, and never generates text.

```bash
hf download notebox/Wisp-v0.1.2-4B --include "wisp-decide/*" --local-dir .
pip install "./wisp-decide[model]"
```

Requires Python 3.10 or newer. The base model is loaded in 4-bit; performance depends on your hardware and configuration. Use the Rust/Candle runtime above for the primary Wisp setup.

```python
import wisp_decide

wisp = wisp_decide.load("notebox/Wisp-v0.1.2-4B")
d = wisp.decide("Mira: he told me \"you are trash\" again, can a mod check?",
                "Does a moderator need to look at this?", ["REVIEW", "NO_ACTION"], task="moderation")
print(d.decision, d.probabilities, d.needs_review)
```

`task` is optional: `moderation` (reported chat message), `username`, `report_validity`. It picks the built-in policy; pass `policy="..."` to supply your own server rules.

## Server

```bash
wisp-decide serve
```

The default address is `http://127.0.0.1:8781`.

Two request formats on the same server.

**Native**, `POST /decide`:

```json
{"task": "moderation", "text": "...", "question": "Does a moderator need to look at this?", "options": ["REVIEW", "NO_ACTION"]}
```

**Jev-style**, `POST /v1/systemone`: the same body as the Jev/TypeSafe protocol, so a client written for Jev can point at this server by changing the base URL.

```json
{"state": "...", "questions": {
  "review":   {"type": "noul", "instructions": "Does a moderator need to look at this?"},
  "category": {"type": "choice", "instructions": "Category?", "criteria": {"INSULT": "personal insult", "THREAT": null, "NONE": "no violation"}}}}
```

`noul` (yes/no) and `choice` are supported. `score` questions are refused with HTTP 422: Wisp was not trained on rating scales. An optional top-level `"task"` selects the built-in policy. Probabilities are not calibrated; `needs_review` is true under 0.7.

`GET /health` reports the model and that no automatic punishment is applied.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
