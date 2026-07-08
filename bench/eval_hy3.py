#!/usr/bin/env python3
# Scored mini-eval of the live Hy3 endpoint, with full per-item response capture.
import json, time, re, random, urllib.request, concurrent.futures, threading
BASE="http://127.0.0.1:8000/v1"; MODEL="hy3"; SEED=42; N=100; WORKERS=4
GSM_MAXTOK=2048; MMLU_MAXTOK=256
GSM_JSONL="/tmp/hy3_gsm8k.jsonl"; MMLU_JSONL="/tmp/hy3_mmlu.jsonl"; RESULTS="/tmp/hy3_eval_results.json"

def chat(prompt, max_tokens, effort=None):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],
          "max_tokens":max_tokens,"temperature":0.0,"top_p":1.0}
    if effort: body["chat_template_kwargs"]={"reasoning_effort":effort}
    req=urllib.request.Request(BASE+"/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req, timeout=700))
    ch=r["choices"][0]; msg=ch["message"]
    return {"content":msg.get("content") or "", "reasoning":msg.get("reasoning_content") or "",
            "finish":ch.get("finish_reason"), "tokens":r["usage"]["completion_tokens"]}

def gsm_gold(a):
    m=re.search(r"####\s*(-?[\d,\.]+)", a); return m.group(1) if m else None
def norm_num(s):
    if s is None: return None
    s=str(s).strip().rstrip(".").replace(",","").replace("$","").replace(" ","")
    try:
        f=float(s); return str(int(f)) if f==int(f) else ("%g"%f)
    except: return s or None
def gsm_extract(t):
    if not t: return None
    m=re.search(r"[Tt]he answer is[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)", t)
    if m: return m.group(1)
    m=re.search(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)", t)
    if m: return m.group(1)
    nums=re.findall(r"-?\$?\s*[\d,]+(?:\.\d+)?", t); return nums[-1] if nums else None
def mmlu_parse(t):
    if not t: return None
    u=t.strip().upper()
    for pat in [r"ANSWER(?:\s+IS)?[:\s]+\(?([A-D])\)?", r"^\s*\(?([A-D])[\.\)\s]", r"\b([A-D])\b"]:
        m=re.search(pat,u)
        if m: return "ABCD".index(m.group(1))
    return None

def run(name, items, solve, grade, jsonl_path):
    res={"correct":0,"parsefail":0,"reqerr":0,"trunc":0}; lock=threading.Lock(); done=[0]
    records=[None]*len(items)
    def one(pair):
        i,it=pair; reqerr=0
        try: resp=solve(it)
        except Exception as e:
            resp={"content":"","reasoning":"","finish":"reqerr:"+repr(e)[:100],"tokens":0}; reqerr=1
        pred, ok, meta = grade(it, resp)
        pf=1 if pred is None else 0; trunc=1 if resp["finish"]=="length" else 0
        records[i]={"i":i,"gold":meta["gold"],"pred":pred,"ok":int(ok),"parsefail":pf,"reqerr":reqerr,
                    "finish":resp["finish"],"tokens":resp["tokens"],"q":meta["q"][:500],
                    "content":resp["content"],"reasoning":resp["reasoning"]}
        with lock:
            done[0]+=1; res["correct"]+=int(ok); res["parsefail"]+=pf; res["reqerr"]+=reqerr; res["trunc"]+=trunc
            if done[0]%10==0: print(f"  {name} {done[0]}/{len(items)} correct={res['correct']} parsefail={res['parsefail']} trunc={res['trunc']}",flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, list(enumerate(items))))
    with open(jsonl_path,"w") as f:
        for r in records:
            if r: f.write(json.dumps(r)+"\n")
    print(f"  [{name}] saved {sum(1 for r in records if r)} records -> {jsonl_path}",flush=True)
    return res

from datasets import load_dataset
print("loading datasets (openai/gsm8k/main/test, cais/mmlu/all/test) ...",flush=True)
g=load_dataset("openai/gsm8k","main",split="test")
gi=list(range(len(g))); random.Random(SEED).shuffle(gi); gi=gi[:N]
g_items=[(g[i]["question"], norm_num(gsm_gold(g[i]["answer"]))) for i in gi]
m=load_dataset("cais/mmlu","all",split="test")
mi=list(range(len(m))); random.Random(SEED).shuffle(mi); mi=mi[:N]
m_items=[(m[i]["question"], m[i]["choices"], m[i]["answer"]) for i in mi]
print(f"gsm8k={len(g_items)} mmlu={len(m_items)} seed={SEED} workers={WORKERS} temp=0 gsm_maxtok={GSM_MAXTOK}",flush=True)

def gsm_solve(it):
    q,_=it
    return chat(q+"\n\nSolve it. End your response with a line exactly: 'The answer is X' (X = the final numeric answer).", GSM_MAXTOK, "high")
def gsm_grade(it, resp):
    q,gold=it
    pred=norm_num(gsm_extract(resp["content"]))
    if pred is None and resp["reasoning"]: pred=norm_num(gsm_extract(resp["reasoning"]))
    return pred, (pred is not None and pred==gold), {"gold":gold,"q":q}
def mmlu_solve(it):
    q,ch,_=it
    opts="\n".join(f"{c}. {t}" for c,t in zip("ABCD",ch))
    return chat(f"{q}\n\n{opts}\n\nRespond with ONLY the single letter of the correct answer (A, B, C, or D).", MMLU_MAXTOK, None)
def mmlu_grade(it, resp):
    q,ch,gold=it
    pred=mmlu_parse(resp["content"])
    if pred is None and resp["reasoning"]: pred=mmlu_parse(resp["reasoning"])
    return pred, (pred is not None and pred==gold), {"gold":gold,"q":q}

t0=time.time()
print("== GSM8K (reasoning_effort=high) ==",flush=True)
gr=run("gsm8k", g_items, gsm_solve, gsm_grade, GSM_JSONL); tg=time.time()-t0
print(f"GSM8K: {gr['correct']}/{N} parsefail={gr['parsefail']} trunc={gr['trunc']} reqerr={gr['reqerr']} ({tg:.0f}s)",flush=True)
t1=time.time()
print("== MMLU (default) ==",flush=True)
mr=run("mmlu", m_items, mmlu_solve, mmlu_grade, MMLU_JSONL); tm=time.time()-t1
print(f"MMLU: {mr['correct']}/{N} parsefail={mr['parsefail']} trunc={mr['trunc']} reqerr={mr['reqerr']} ({tm:.0f}s)",flush=True)
wall=time.time()-t0
out={"gsm8k":{"correct":gr['correct'],"n":N,"pct":round(100*gr['correct']/N,1),"parsefail":gr['parsefail'],"trunc":gr['trunc'],"reqerr":gr['reqerr'],"sec":round(tg)},
     "mmlu":{"correct":mr['correct'],"n":N,"pct":round(100*mr['correct']/N,1),"parsefail":mr['parsefail'],"trunc":mr['trunc'],"reqerr":mr['reqerr'],"sec":round(tm)},
     "seed":SEED,"workers":WORKERS,"temperature":0.0,"gsm_maxtok":GSM_MAXTOK,"wall_sec":round(wall),
     "datasets":{"gsm8k":"openai/gsm8k/main/test","mmlu":"cais/mmlu/all/test"},"endpoint":BASE,"model":MODEL}
json.dump(out, open(RESULTS,"w"), indent=2)
print("=== DONE ===",flush=True); print(json.dumps(out,indent=2),flush=True)
