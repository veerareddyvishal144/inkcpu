#!/usr/bin/env python3
"""Write the 8 DSpark acceptance prompts to tools/dspark/prompts/<name>.txt (raw prompt text with Inkling special markers,
see docs/ARCH.md section 4 and plan/raw/hf/repo/chat_template.jinja). Run: /usr/bin/python3 -I make_prompts.py"""
import os, re, glob, json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
os.makedirs(OUT, exist_ok=True)

def eff(x):  # the template's thinking-effort system message
    return "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>" % x

def user(t):
    return "<|message_user|><|content_text|>%s<|end_message|>" % t

P = {}
# 1 chat answer (no thinking)
P["chat_answer"] = eff(0) + user("Explain in one short paragraph why the sky is blue during the day and red at sunset.") + "<|message_model|><|content_text|>"
# 2 code completion (raw text, no markers)
P["code_completion"] = '''import heapq
from collections import defaultdict


def dijkstra(graph, source):
    """Return the shortest distance from source to every reachable node.

    graph is an adjacency dict: graph[u] = [(v, weight), ...] with non-negative weights.
    """
    dist = defaultdict(lambda: float("inf"))
    dist[source] = 0
    heap = [(0, source)]
    while heap:
'''
# 3 prose continuation (raw text)
P["prose_continuation"] = ("The old lighthouse keeper had not spoken to another living soul in nearly three weeks when the storm finally "
                           "cleared. The sea, grey and heaving, slowly settled back into its long habit of rolling toward the rocks, and far out "
                           "on the horizon, where the clouds had begun to break into ragged strips of pale gold, he saw the first sail.")
# 4 reasoning-mode turn (thinking prefilled)
P["reasoning"] = eff(0.9) + user("A train leaves city A at 9:00 travelling toward city B at 80 km/h. City B is 300 km away. A second train leaves B at 9:30 "
                                 "travelling toward A at 100 km/h. At what time and how far from A do they meet?") + "<|message_model|><|content_thinking|>"
# 5 tool-call turn
tools = [{"description": "Get the current weather for a city", "name": "get_weather",
          "parameters": {"properties": {"city": {"description": "City name", "type": "string"},
                                        "unit": {"enum": ["celsius", "fahrenheit"], "type": "string"}},
                         "required": ["city"], "type": "object"}, "type": "function"}]
P["tool_call"] = ("<|message_system|>tool_declare<|content_xml|>" + json.dumps(tools, sort_keys=True, separators=(",", ":")) + "<|end_message|>"
                  + eff(0) + user("What is the weather like in Lisbon right now? Please use celsius.") + "<|message_model|>")
# 6 ~500-token context question
wiki = None
for p in glob.glob("/data/inkling-mini/wiki/**/*.raw", recursive=True) + glob.glob("/data/inkling-mini/wiki/**/*.txt", recursive=True):
    wiki = open(p, encoding="utf-8", errors="replace").read(); break
if wiki is None:
    raise SystemExit("wikitext not found")
body = wiki[:2600]
body = body.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").replace(" , ", ", ").replace(" . ", ". ").replace(" ; ", "; ")
body = re.sub(r"\s*\n\s*", "\n", body).strip()
cut = body.rfind(". ") + 1
body = body[:cut]
P["long_context_qa"] = eff(0) + user("Read the following passage and then answer the question.\n\nPassage:\n" + body +
                                     "\n\nQuestion: Summarise the passage above in three sentences.") + "<|message_model|><|content_text|>"
# 7 second chat answer (writing task)
P["chat_email"] = eff(0) + user("Write a short, friendly email to a colleague asking to move our Tuesday project meeting to Thursday afternoon.") + "<|message_model|><|content_text|>"
# 8 code in chat
P["chat_code"] = eff(0) + user("Write a Python function is_palindrome(s) that ignores case and punctuation, with a short docstring and two example calls.") + "<|message_model|><|content_text|>"

for k, v in P.items():
    with open(os.path.join(OUT, k + ".txt"), "w", encoding="utf-8") as f:
        f.write(v)
    print(k, len(v), "chars")
