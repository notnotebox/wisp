from __future__ import annotations

import argparse
import json

from .engine import load
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="wisp-decide", description="Wisp v0.1.2 4B decisions")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "ask"):
        p = sub.add_parser(name)
        p.add_argument("--adapter", default="notebox/Wisp-v0.1.2-4B", help="Hub id or local folder")
        p.add_argument("--no-quantize", action="store_true", help="load the base in bf16 instead of 4-bit")
    sub.choices["serve"].add_argument("--host", default="127.0.0.1")
    sub.choices["serve"].add_argument("--port", type=int, default=8781)
    ask = sub.choices["ask"]
    ask.add_argument("text")
    ask.add_argument("--question", default="Does a moderator need to look at this?")
    ask.add_argument("--options", nargs="+", default=["REVIEW", "NO_ACTION"])
    ask.add_argument("--task", choices=["moderation", "username", "report_validity"])
    args = parser.parse_args()
    wisp = load(args.adapter, quantize=not args.no_quantize)
    if args.command == "serve":
        serve(wisp, args.host, args.port)
    else:
        print(json.dumps(wisp.decide(args.text, args.question, args.options, task=args.task).as_dict(),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
