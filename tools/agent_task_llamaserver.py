import json,urllib.request,subprocess,time,os,sys
W="/data/inkling-mini/agent-work"
TOOLS=[{"type":"function","function":{"name":n,"description":d,"parameters":{"type":"object","properties":p,"required":list(p)}}} for n,d,p in [
 ("read_file","Read a file in the work dir",{"path":{"type":"string"}}),
 ("write_file","Write a file in the work dir",{"path":{"type":"string"},"content":{"type":"string"}}),
 ("run_tests","Run python3 -m unittest -v in the work dir",{})]]
def tool(n,a):
    try:
        if n=="run_tests":
            r=subprocess.run(["python3","-I","-m","unittest","-v"],cwd=W,capture_output=True,text=True,timeout=60,env={"PATH":"/usr/bin:/bin","PYTHONPATH":W})
            return (r.stdout+r.stderr)[-2500:]
        p=os.path.realpath(os.path.join(W,a["path"]))
        if not p.startswith(W+"/"): return "error: path outside work dir"
        if n=="read_file": return open(p).read()[:6000]
        if n=="write_file": open(p,"w").write(a["content"]); return "ok"
    except Exception as e: return f"error: {e}"
    return "error: unknown tool"
msgs=[{"role":"system","content":"You are a coding agent. Use the tools. Finish when all tests pass."},
 {"role":"user","content":"Implement textstats.py in the work dir so that test_textstats.py passes. Read the tests first, write the module, run the tests, and fix failures."}]
T0=time.time()
for step in range(1,13):
    t=time.time()
    req=urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",json.dumps({"messages":msgs,"tools":TOOLS,"max_tokens":3000,"temperature":0.6}).encode(),{"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=3000)); m=d["choices"][0]["message"]; tm=d["timings"]
    calls=m.get("tool_calls") or []
    print(f"[step {step}] {time.time()-t:.0f}s prompt_n={tm['prompt_n']} cached={tm.get('cache_n')} gen={tm['predicted_n']}@{tm['predicted_per_second']:.1f} calls={[c['function']['name'] for c in calls]}",flush=True)
    msgs.append({k:v for k,v in m.items() if k in("role","content","tool_calls")})
    if not calls: print("FINAL:",(m.get("content") or "")[:400],flush=True); break
    for c in calls:
        try: a=json.loads(c["function"]["arguments"] or "{}")
        except Exception: a={}
        out=tool(c["function"]["name"],a)
        print(f"   -> {c['function']['name']}: {out[:150]!r}",flush=True)
        msgs.append({"role":"tool","tool_call_id":c["id"],"content":out})
print("TOTAL",f"{time.time()-T0:.0f}s",flush=True)
print(tool("run_tests",{})[-400:],flush=True)
