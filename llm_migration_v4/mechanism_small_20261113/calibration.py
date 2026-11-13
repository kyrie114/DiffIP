"""只使用已完成缓存：比较绝对阈值、同提示相减、减常数、错位提示相减。
所有阈值由种子组0的留出无关模型最大值确定；组1/2检验。"""
import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
R=json.loads((H/'results.json').read_text());CACHE={p.stem:json.loads(p.read_text()) for p in (H/'cache').glob('*.json')}
POS=['instruct','instruct_prune'];out={}
for m,mr in R['metrics'].items():
    out[m]={}
    for mode in ['absolute','same_prompt','subtract_constant','wrong_prompt']:
        dirs=[]
        for d in mr['directions']:
            ref=np.array(CACHE[d['reference']][m]); test=np.array(CACHE[d['heldout']][m]); ps={n:np.array(CACHE[n][m]) for n in POS}
            if mode=='absolute': baseline=np.zeros_like(ref)
            elif mode=='same_prompt':baseline=ref
            elif mode=='subtract_constant':baseline=np.full_like(ref,ref[:,0].mean())
            else:baseline=np.roll(ref,1,axis=0)
            t=float((test[:,0]-baseline[:,0]).max());seeds={}
            for sd in [1,2]:seeds[str(sd)]={'fpr':float(np.mean(test[:,sd]-baseline[:,sd]>=t)),'tpr':{n:float(np.mean(p[:,sd]-baseline[:,sd]>=t)) for n,p in ps.items()}}
            ok=all(s['fpr']<=.1 and min(s['tpr'].values())>=.9 for s in seeds.values())
            dirs.append(dict(reference=d['reference'],heldout=d['heldout'],threshold=t,seeds=seeds,passed=ok))
        out[m][mode]={'pass_count':sum(d['passed'] for d in dirs),'directions':dirs}
(H/'calibration.json').write_text(json.dumps(out,indent=1))
for m,v in out.items():print(m,{k:r['pass_count'] for k,r in v.items()})
