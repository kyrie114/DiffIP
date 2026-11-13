"""新版二合一方案的小模型机理检验。仅复用实采缓存，无新增模型前向。
运行：python3 llm_migration_v4/mechanism_small_20261113/run.py
32条提示，种子123/1123/2123，128次，均匀4层，PCA8。
静态2×2：同四层、TruthfulQA前128项，隔离池化/层位以外的混杂。
扰动指标：原始CKA、逐维标准化CKA、原ALS、打乱配对、强缩放。
每个模型单独缓存；固定阈值仅用种子组0；组1/2留出，无测试集调参。
"""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import json, math, sys, time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent; ROOT=H.parents[1]; V3=ROOT/'llm_migration_v3'; SA=ROOT/'llm_migration_v4/small_adv'
sys.path.insert(0,str(ROOT))
from diffip.preprocessing import project_sequences
from diffip.reversion import fit_step_alignment
NEG=['neg_smollm2_135m','neg_smollm2_360m','neg_gpt2','neg_pythia_160m','neg_qwen2_0.5b']
POS=['instruct','instruct_prune']; EXTRA=['width75h','vocab_swap']
OUT=H/'cache'; OUT.mkdir(exist_ok=True)
def ld(n,static=False):
    if static:
        z=np.load(V3/'reef_baseline/acts'/f'{n}.npz'); a=z['tqa200_last']; L=[round(i*(a.shape[0]-1)/3) for i in range(4)]; return a[L,:128].astype(float)
    p=SA/'feats'/f'pert_{n}.npz'
    if not p.exists(): p=V3/'feats'/f'pert_{n}.npz'
    return np.load(p)['pert']
def cka(x,y,standard=False):
    x=x-x.mean(0); y=y-y.mean(0)
    if standard:
        sx=x.std(0,ddof=1); sy=y.std(0,ddof=1); sx[sx==0]=1; sy[sy==0]=1; x=x/sx; y=y/sy
    a=x@x.T; b=y@y.T
    return float(np.sum(a*b)/max(np.sqrt(np.sum(a*a)*np.sum(b*b)),1e-30))
def als(x,y):
    np.random.seed(2026)
    pr=project_sequences(x,y,8)
    return math.exp(-np.mean([fit_step_alignment(a,b).distance for a,b in zip(pr.suspect,pr.victim)]))
def auc(p,n):
    p=np.array(p).ravel();n=np.array(n).ravel();d=p[:,None]-n[None,:];return float(np.mean(d>0)+.5*np.mean(d==0))
def run():
    v=ld('victim'); result={};t0=time.time()
    for n in POS+NEG+EXTRA:
        f=OUT/f'{n}.json'
        if f.exists(): result[n]=json.loads(f.read_text());print('cached',n,flush=True);continue
        s=ld(n);D=s.shape[-1];rng=np.random.default_rng(2026);scale=np.exp(rng.uniform(-4,4,D));perm=rng.permutation(128)
        r={k:np.zeros((32,3)) for k in ['als','cka','cka_std','als_shuffle','als_scale','cka_scale','cka_std_scale']}
        for p in range(32):
            for k in range(3):
                x=s[p,k].astype(float);y=v[p,k].astype(float)
                r['als'][p,k]=als(x,y)
                r['als_shuffle'][p,k]=als(x[:,perm,:],y)
                r['als_scale'][p,k]=als(x*scale,y)
                for std,key in [(False,'cka'),(True,'cka_std')]:
                    r[key][p,k]=np.mean([cka(a,b,std) for a,b in zip(x,y)])
                    r[key+'_scale'][p,k]=np.mean([cka(a*scale,b,std) for a,b in zip(x,y)])
            if p%8==7: print(n,p+1,'/32',round(time.time()-t0,1),'s',flush=True)
        r={key:a.tolist() for key,a in r.items()}
        if n not in EXTRA:
            x=ld(n,True);y=ld('victim',True)
            r['static_matched']={'als':als(x,y),'cka':float(np.mean([cka(a,b) for a,b in zip(x,y)])), 'cka_std':float(np.mean([cka(a,b,True) for a,b in zip(x,y)]))}
        f.write_text(json.dumps(r));result[n]=r
        del s
    summary={}
    for n,r in result.items(): summary[n]={key:float(np.mean(a)) for key,a in r.items() if key!='static_matched'};summary[n]['static_matched']=r.get('static_matched')
    # Compare clean and scaled positive derivatives without selecting successful seeds.
    metrics={}
    for m in ['als','cka','cka_std']:
        neg=np.stack([result[n][m] for n in NEG]); pos=np.stack([result[n][m] for n in POS]); sc=np.stack([result[n][m+'_scale'] for n in POS])
        metrics[m]={'clean_auc':auc(pos,neg),'scaled_auc':auc(sc,neg), 'max_negative_model_mean':float(neg.mean((1,2)).max())}
        dirs=[]
        for i,ref in enumerate(NEG):
            for j,test in enumerate(NEG):
                if i==j:continue
                a,b=neg[i],neg[j]; threshold=float((b[:,0]-a[:,0]).max()); by={}
                for sd in (1,2):
                    by[str(sd)]={'fpr':float(np.mean(b[:,sd]-a[:,sd]>=threshold)),
                                'tpr':{n:float(np.mean(np.array(result[n][m])[:,sd]-a[:,sd]>=threshold)) for n in POS+EXTRA},
                                'scaled_tpr':{n:float(np.mean(np.array(result[n][m+'_scale'])[:,sd]-a[:,sd]>=threshold)) for n in POS}}
                passed=all(by[str(sd)]['fpr']<=.1 and all(t>=.9 for t in by[str(sd)]['tpr'].values()) for sd in (1,2))
                corepass=all(by[str(sd)]['fpr']<=.1 and all(by[str(sd)]['tpr'][n]>=.9 for n in POS) for sd in (1,2))
                dirs.append(dict(reference=ref,heldout=test,threshold=threshold,seeds=by,pass_all=passed,pass_core=corepass))
        metrics[m]['directions']=dirs;metrics[m]['pass_core_count']=sum(d['pass_core'] for d in dirs);metrics[m]['pass_all_count']=sum(d['pass_all'] for d in dirs)
    # Exact algebraic counterexample: general invertible shear vs restricted rotation+diagonal scaling.
    rng=np.random.default_rng(88);x=rng.normal(size=(128,8));A=np.eye(8);A[0,1]=3;A[2,5]=-2;y=x@A
    old=fit_step_alignment(x,y).distance;xc=x-x.mean(0);yc=y-y.mean(0);B=np.linalg.lstsq(xc,yc,rcond=None)[0];full=float(np.linalg.norm(xc@B-yc,axis=1).mean())
    res={'source_commit':'7f8c5f5','summary':summary,'metrics':metrics,'shear_counterexample':{'restricted_als_distance':old,'full_affine_mean_residual':full},'elapsed_seconds':time.time()-t0}
    (H/'results.json').write_text(json.dumps(res,indent=1));print(json.dumps({k:{a:b for a,b in r.items() if a!='directions'} for k,r in metrics.items()},indent=1),flush=True);print(res['shear_counterexample'],flush=True)
if __name__=='__main__':run()
