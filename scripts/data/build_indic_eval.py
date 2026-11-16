import json, re, collections
rows=[json.loads(l) for l in open('data/cands/ml_gsm8k_indic.jsonl')]
print(list(rows[0].keys()))
out=[]; c=collections.Counter()
for r in rows:
    lang=r['_cfg']
    if lang=='en': continue
    a=r.get('answer') or r.get('original_answer') or ''
    m=re.search(r'####\s*([-\d.,]+)', str(a)); 
    if not m: 
        nums=re.findall(r'-?\d[\d,]*(?:\.\d+)?', str(r.get('original_answer') or a)); 
        if not nums: continue
        ans=nums[-1]
    else: ans=m.group(1)
    q=r.get('question') or r.get('translated_question')
    c[lang]+=1
    out.append({'id':f'gsmindic-{lang}-{c[lang]}', 'lang':lang, 'question':q, 'answer':ans.replace(',','').rstrip('.')})
with open('data/eval_gsm8k_indic.jsonl','w') as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
print(c, json.dumps(out[0],ensure_ascii=False)[:400])
