"""
Performance Metrics — timers, speedup, efficiency, throughput, memory.
"""
import time, os, json, numpy as np

class PerfTimer:
    def __init__(self): self._s={}; self._e={}
    def start(self,n): self._s[n]=time.perf_counter()
    def stop(self,n):
        if n in self._s: dt=time.perf_counter()-self._s[n]; self._e[n]=self._e.get(n,0)+dt; del self._s[n]; return dt
        return 0
    def elapsed(self,n): return self._e.get(n,0)
    def record(self,n,v): self._e[n]=v
    def get(self,n,d=0): return self._e.get(n,d)

def compute_speedup(ts,tp): return ts/tp if tp>0 else float("inf")
def compute_efficiency(sp,p): return sp/p if p>0 else 0
def compute_eff_loss(e): return 1-e
def compute_throughput(ns,ne,t): return (ns*ne)/t if t>0 else float("inf")

def get_memory_mb():
    try:
        import resource,platform
        u=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return u/(1024*1024) if platform.system()=="Darwin" else u/1024
    except:
        try: import psutil; return psutil.Process(os.getpid()).memory_info().rss/(1024*1024)
        except: return 0

def aggregate(all_m, args):
    mt=max(m["train_time_s"] for m in all_m)
    ac=np.mean([m["compute_time_s"] for m in all_m])
    am=np.mean([m["comm_time_s"] for m in all_m])
    ai=np.mean([m["io_time_s"] for m in all_m])
    amem=np.mean([m["memory_used_mb"] for m in all_m])
    ds=all_m[0].get("dataset_samples",0)
    # Sum comm bytes across all ranks (total inter-rank data moved)
    total_comm_bytes_mb=sum(m.get("comm_bytes_mb",0) for m in all_m)
    return {"strategy":args.strategy,"mpi_ranks":len(all_m),
        "threads_per_rank":all_m[0]["threads_per_rank"],"total_workers":all_m[0]["total_workers"],
        "dataset_samples":ds,"dataset_size_mb":all_m[0].get("dataset_size_mb",0),
        "batch_size":args.batch_size,"epochs":args.epochs,"io_mode":args.io_mode,
        "train_time_s":mt,"compute_time_s":ac,"comm_time_s":am,"io_time_s":ai,
        "compute_frac":ac/max(mt,1e-9),"comm_frac":am/max(mt,1e-9),"io_frac":ai/max(mt+ai,1e-9),
        "throughput":compute_throughput(ds,args.epochs,mt),
        "epochs_per_second":args.epochs/max(mt,1e-9),
        "total_comm_bytes_mb":total_comm_bytes_mb,
        "avg_mem_mb":amem,"per_rank_mem":[m["memory_used_mb"] for m in all_m],
        "final_loss":all_m[0]["final_loss"],"epoch_losses":all_m[0].get("epoch_losses",[]),
        "loss_improvement": (all_m[0].get("epoch_losses",[None,None])[0] or 0) - (all_m[0].get("final_loss") or 0)}

def save_json(fp, summary, all_m):
    with open(fp,"w") as f: json.dump({"summary":_s(summary),"per_rank":[_s(m) for m in all_m]},f,indent=2)

def _s(d):
    r={}
    for k,v in d.items():
        if isinstance(v,np.ndarray): r[k]=v.tolist()
        elif isinstance(v,(np.integer,)): r[k]=int(v)
        elif isinstance(v,(np.floating,)): r[k]=float(v)
        elif isinstance(v,dict): r[k]=_s(v)
        elif isinstance(v,list): r[k]=[float(x) if isinstance(x,(np.floating,)) else int(x) if isinstance(x,(np.integer,)) else x for x in v]
        else: r[k]=v
    return r
