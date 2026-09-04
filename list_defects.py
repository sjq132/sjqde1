import os
from collections import Counter

root="data/raw/mvtec_ad"
result={}
for obj in sorted(os.listdir(root)):
    p=os.path.join(root,obj)
    if not os.path.isdir(p):
        continue
    defects=set()
    for r,d,fs in os.walk(p):
        parts=os.path.normpath(r).split(os.sep)
        tail=parts[-1]
        if tail in ["train","test","good","ground_truth"] or tail==obj:
            continue
        defects.add(tail)
    result[obj]=sorted(defects)

for obj,defs in result.items():
    print(f"{obj}: {defs}")
