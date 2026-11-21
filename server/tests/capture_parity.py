#!/usr/bin/env python
"""Dev tool: capture llama-server (port 8001, --jinja) tokenisation and templating into fixtures/parity.json."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_cases import TEMPLATE_CASES, TOKENIZE_STRINGS  # noqa: E402

BASE = "http://127.0.0.1:8001"


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:500]}


def main():
    out = {"tokenize": [], "template": {}}
    for sx in TOKENIZE_STRINGS:
        out["tokenize"].append({
            "text": sx,
            "special": post("/tokenize", {"content": sx, "add_special": False, "parse_special": True}).get("tokens"),
            "plain": post("/tokenize", {"content": sx, "add_special": False, "parse_special": False}).get("tokens")})
    for name, c in TEMPLATE_CASES.items():
        body = {"messages": c["messages"]}
        if c.get("tools"):
            body["tools"] = c["tools"]
        e = c.get("reasoning_effort")
        if e is not None:
            body["reasoning_effort"] = e
            body["chat_template_kwargs"] = {"reasoning_effort": e}
        r = post("/apply-template", body)
        rec = {"llama_prompt": r.get("prompt"), "error": r.get("error")}
        if r.get("prompt") is not None:
            rec["llama_ids"] = post("/tokenize", {"content": r["prompt"], "add_special": False, "parse_special": True}).get("tokens")
        out["template"][name] = rec
        print(name, "ERR " + str(r.get("error")) if r.get("error") else "ok %d chars" % len(r["prompt"]), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
