import os
from collections import Counter

c = Counter()
paths = []
for r, d, fs in os.walk("data/processed/labels"):
    for f in fs:
        if f.endswith(".txt"):
            p = os.path.join(r, f)
            if os.path.getsize(p) > 0:
                paths.append(p)

for p in paths:
    first = open(p, encoding="utf-8").read().splitlines()[0].split()[0]
    c[first] += 1

print("总非空标签数:", len(paths))
print("类别分布:", dict(sorted(c.items())))

names = {
    "0": "good",
    "1": "crack",
    "2": "scratch",
    "3": "broken",
    "4": "contamination",
    "5": "missing_deform",
    "6": "color",
    "7": "hole_poke",
    "8": "glue_cut",
    "9": "other_defect",
}
for k, v in sorted(c.items()):
    print(k, names.get(k, "?"), v)