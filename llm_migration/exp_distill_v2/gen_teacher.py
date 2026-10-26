"""步骤1：教师 Qwen2.5-0.5B-Instruct 在 Alpaca 指令上贪心生成回答（序列级蒸馏数据）。
同时保存 Alpaca 原始回答（text-davinci-003 生成，非教师）作为对照学生的训练文本。CPU fp32。"""
import json, sys, time, random, torch, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(8); random.seed(0)
N_TRAIN, N_HELD = int(sys.argv[1]), int(sys.argv[2]); MAXNEW = 128; BS = 16
rows = pq.read_table(hf_hub_download("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet", repo_type="dataset")).to_pylist()
rows = [r for r in rows if not r["input"].strip() and 20 <= len(r["output"]) <= 600]
random.shuffle(rows); rows = rows[:N_TRAIN + N_HELD]
T = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(T); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(T, dtype=torch.float32).eval()
out = []; t0 = time.time()
with torch.no_grad():
    for i in range(0, len(rows), BS):
        b = rows[i:i+BS]
        ps = [tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True) for r in b]
        enc = tok(ps, return_tensors="pt", padding=True)
        g = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, s in zip(b, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out.append(dict(instruction=r["instruction"], teacher=s.strip(), alpaca=r["output"].strip()))
        print(f"{len(out)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        if len(out) % 160 == 0 or len(out) == len(rows):
            json.dump(dict(train=out[:N_TRAIN], heldout=out[N_TRAIN:]), open("data/teacher_data.json", "w"), ensure_ascii=False, indent=0)
