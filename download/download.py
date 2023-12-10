import ssgetpy
import sys

start = int(sys.argv[1])
for i in range(int(sys.argv[1]), min(start+200,2894)):
    ssgetpy.fetch(name_or_id=i, limit=10,location=f"/home/lzy/spmm/data_{start}_{start+200}", format="MM")
    print(f"[INFO] Download {i} / 2893")   
