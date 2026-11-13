import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=json.loads((H/'results.json').read_text());S=R['summary']
fig,axs=plt.subplots(1,2,figsize=(11,4))
names=['instruct','instruct_prune','width75h','vocab_swap']; labels=['Instruct','Drop 4 layers','Width 75% + heal','Vocab replaced']
for j,(m,lbl) in enumerate([('als','Original alignment'),('cka','Raw CKA'),('cka_std','Standardized CKA')]):
    vals=[S[n][m] for n in names];axs[0].bar(np.arange(4)+.25*(j-1),vals,.24,label=lbl)
axs[0].set_xticks(range(4),labels,rotation=15);axs[0].set_ylabel('Mean similarity (not shared scale)');axs[0].legend(fontsize=7)
x=np.arange(3)
for j,(field,lbl) in enumerate([('als','Clean'),('als_shuffle','Pairing broken'),('als_scale','Anisotropic scale')]):
    axs[1].bar(x+(j-1)*.25,[S[n][field] for n in ['instruct','instruct_prune','neg_qwen2_0.5b']],.24,label=lbl)
axs[1].set_xticks(x,['Instruct','Drop 4 layers','Unrelated Qwen2']);axs[1].set_ylabel('Original alignment similarity');axs[1].legend(fontsize=7)
fig.suptitle('Qwen2.5-0.5B | 32 prompts x 3 seeds | measured cached activations');fig.tight_layout();fig.savefig(H/'comparison.png',dpi=180)
