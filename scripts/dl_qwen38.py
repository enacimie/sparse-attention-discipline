import json, os, sys, urllib.request, concurrent.futures, time

REPO = "Qwen/Qwen3.8-27B"
DEST = "/root/qwen38"
os.makedirs(DEST, exist_ok=True)

t0 = time.time()
with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}") as r:
    files = [f["rfilename"] for f in json.load(r)["siblings"]]
print(f"{len(files)} archivos a descargar", flush=True)

done = [0]
def dl(fn):
    url = f"https://huggingface.co/{REPO}/resolve/main/{fn}"
    dst = os.path.join(DEST, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        done[0] += 1
        return f"skip {fn}"
    os.makedirs(os.path.dirname(dst) or DEST, exist_ok=True)
    urllib.request.urlretrieve(url, dst)
    done[0] += 1
    return f"done {fn} ({done[0]}/{len(files)})"

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    for r in ex.map(dl, files):
        print(r, flush=True)

print(f"TOTAL: {len(files)} archivos en {time.time()-t0:.0f}s", flush=True)
