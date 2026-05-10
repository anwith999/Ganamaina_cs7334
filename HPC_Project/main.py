#!/usr/bin/env python3
"""
Parallel NN Training: Data Parallel vs Model Parallel
PyTorch | MPI + OpenMP | Regression (MSE) | HDF5 | Caliper | PAPI
"""
import argparse, os, numpy as np, h5py
from mpi4py import MPI
from train_sequential import train_sequential
from train_data_parallel import train_data_parallel
from train_model_parallel import train_model_parallel
from perf_metrics import PerfTimer, get_memory_mb, aggregate, save_json
from papi_counters import PAPI
from caliper_wrapper import caliper_region, caliper_available

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--strategy",default="data_parallel",choices=["sequential","data_parallel","model_parallel"])
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch_size",type=int,default=256)
    p.add_argument("--hidden_dim",type=int,default=512)
    p.add_argument("--n_hidden",type=int,default=4)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--io_mode",default="parallel",choices=["parallel","broadcast"])
    p.add_argument("--data_file",default="data/dataset_small.h5")
    p.add_argument("--output_dir",default="results")
    p.add_argument("--enable_papi",action="store_true")
    return p.parse_args()

def main():
    comm=MPI.COMM_WORLD; rank,size=comm.Get_rank(),comm.Get_size()
    args=parse_args(); nt=int(os.environ.get("OMP_NUM_THREADS",1))

    with h5py.File(args.data_file,"r") as f:
        ns,nf,ntarg=f["X"].shape[0],f["X"].shape[1],f["y"].shape[1]
        src=f.attrs.get("source","unknown")
    ds_mb=os.path.getsize(args.data_file)/(1024*1024)

    if rank==0:
        os.makedirs(args.output_dir,exist_ok=True)
        print("="*70)
        print("  PARALLEL NN TRAINING: DATA PARALLEL vs MODEL PARALLEL")
        print("  Task: REGRESSION (MSE Loss) | Scientific Data")
        print("="*70)
        for k,v in [("Strategy",args.strategy),("MPI Ranks",size),("OMP Threads",nt),
                      ("Epochs",args.epochs),("Batch Size",args.batch_size),
                      ("Model",f"{args.n_hidden} hidden x {args.hidden_dim}"),
                      ("Dataset",f"{args.data_file} ({ns:,} x {nf} -> {ntarg}, {ds_mb:.0f}MB)"),
                      ("Source",src),("I/O Mode",args.io_mode),
                      ("Caliper","ON" if caliper_available() else "off"),
                      ("PAPI","ON" if args.enable_papi else "off")]:
            print(f"  {k:<14}: {v}")
        print("="*70)

    papi=PAPI() if args.enable_papi else None
    timer=PerfTimer()

    mem0=get_memory_mb()
    timer.start("train_total")
    with caliper_region("full_run"):
        if args.strategy=="sequential": res=train_sequential(args,timer)
        elif args.strategy=="data_parallel": res=train_data_parallel(args,comm,rank,size,timer)
        elif args.strategy=="model_parallel": res=train_model_parallel(args,comm,rank,size,timer)
    timer.stop("train_total")
    mem1=get_memory_mb()

    if papi: papi.stop(); papi.print_report(rank)

    lm={"rank":rank,"strategy":args.strategy,"mpi_ranks":size,"threads_per_rank":nt,
        "total_workers":size*nt,"dataset_samples":ns,"dataset_size_mb":ds_mb,
        "batch_size":args.batch_size,"epochs":args.epochs,"io_mode":args.io_mode,
        "io_time_s":timer.get("io_load",0),"train_time_s":timer.elapsed("train_total"),
        "compute_time_s":timer.get("compute",0),"comm_time_s":timer.get("communication",0),
        "memory_before_mb":mem0,"memory_after_mb":mem1,"memory_used_mb":mem1-mem0,
        "epoch_losses":res.get("epoch_losses",[]),"final_loss":res.get("final_loss"),
        "comm_bytes_mb":res.get("comm_bytes_mb",0.0),
        "papi":papi.report() if papi else None}

    am=comm.gather(lm,root=0)
    if rank==0:
        s=aggregate(am,args)
        print("\n"+"="*70+"\n  RESULTS\n"+"="*70)
        for k,v in [("Train Time",f"{s['train_time_s']:.4f}s"),
                      ("Epochs/s",f"{s['epochs_per_second']:.4f}"),
                      ("Compute",f"{s['compute_time_s']:.4f}s ({s['compute_frac']:.1%})"),
                      ("Communication",f"{s['comm_time_s']:.4f}s ({s['comm_frac']:.1%})"),
                      ("Comm Data",f"{s['total_comm_bytes_mb']:.2f} MB transferred"),
                      ("I/O",f"{s['io_time_s']:.4f}s ({s['io_frac']:.1%})"),
                      ("Throughput",f"{s['throughput']:.0f} samples/s"),
                      ("Memory",f"{s['avg_mem_mb']:.1f} MB"),
                      ("Final MSE",f"{s['final_loss']:.6f}" if s['final_loss'] else "N/A"),
                      ("Loss Improve",f"{s['loss_improvement']:.6f}"),
                      ("Mem/Batch",f"{s['avg_mem_mb']/max(args.batch_size,1):.4f} MB")]:
            print(f"  {k:<14}: {v}")
        print("="*70)
        of=os.path.join(args.output_dir,f"metrics_{args.strategy}_{size}r_{nt}t_{ns}s.json")
        save_json(of,s,am); print(f"  Saved: {of}")
    comm.Barrier()
    if rank==0: print("\nDone.")

if __name__=="__main__": main()
