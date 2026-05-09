"""
Strategy 1: Data Parallel (PyTorch DistributedDataParallel)
- Full model replica on each rank
- Data split via DistributedSampler / parallel I/O
- Gradients averaged via Allreduce (handled by DDP)
- MPI used for process group init + explicit timing
- OpenMP threads for BLAS operations
"""
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time, os, numpy as np
from mpi4py import MPI
from model import RegressionNN
from dataset import load_parallel_io, load_broadcast, make_loader
from caliper_wrapper import caliper_region

def train_data_parallel(args, comm, rank, size, timer):
    # Init torch.distributed with gloo backend
    # MASTER_ADDR defaults to localhost for single-node; set via env for multi-node
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(size)})
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=rank, world_size=size)

    nt = int(os.environ.get("OMP_NUM_THREADS",1))
    torch.set_num_threads(nt)

    # Load data using chosen I/O strategy
    with caliper_region("data_io"):
        timer.start("io_load")
        if args.io_mode == "parallel":
            ds = load_parallel_io(args.data_file, rank, size)
        else:
            ds = load_broadcast(args.data_file, rank, size, comm)
        timer.stop("io_load")

    loader = make_loader(ds, args.batch_size)
    if rank == 0: print(f"  Rank 0: {len(ds)} local samples")

    model = RegressionNN(ds.n_features, args.hidden_dim, ds.n_targets, args.n_hidden)
    ddp = DDP(model)
    crit = nn.MSELoss(); opt = torch.optim.SGD(ddp.parameters(), lr=args.lr)

    losses = []
    comp_t, comm_t = 0.0, 0.0
    total_batches = 0

    # Bytes per DDP gradient allreduce = param count × 4 bytes (float32)
    param_bytes = sum(p.numel() * 4 for p in model.parameters())

    with caliper_region("dp_training"):
        for ep in range(args.epochs):
            el, nb = 0, 0
            for Xb, yb in loader:
                t0 = time.perf_counter()
                opt.zero_grad(); pred = ddp(Xb); loss = crit(pred, yb)
                loss.backward()
                opt.step()
                comp_t += time.perf_counter() - t0
                el += loss.item(); nb += 1

            # Explicit allreduce to synchronize loss for logging
            t0 = time.perf_counter()
            lt = torch.tensor([el, float(nb)])
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            comm_t += time.perf_counter() - t0

            avg = lt[0].item() / lt[1].item()
            losses.append(avg)
            total_batches += nb
            if rank == 0: print(f"  [DataParallel] Epoch {ep+1}/{args.epochs} | MSE: {avg:.6f}")

    # Actual bytes communicated:
    #   DDP allreduce per batch: param_bytes per rank (ring-allreduce sends ~param_bytes total)
    #   Loss allreduce per epoch: 2 floats × 4 bytes = 8 bytes
    comm_bytes = param_bytes * total_batches + 8 * args.epochs
    comm_bytes_mb = comm_bytes / (1024 * 1024)

    timer.record("compute", comp_t); timer.record("communication", comm_t)
    dist.destroy_process_group()
    return {"epoch_losses": losses, "final_loss": losses[-1], "comm_bytes_mb": comm_bytes_mb}
