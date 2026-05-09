"""
Strategy 2: Model Parallel (Layer-wise Split via MPI)
- Layers split across MPI ranks
- Forward: send activations rank i -> rank i+1 (MPI.Send)
- Backward: send gradients rank i+1 -> rank i (MPI.Send)
- Each rank updates only its own parameters
- Uses raw mpi4py for communication (explicit MPI usage)
- OpenMP threads for BLAS operations
"""
import torch, torch.nn as nn, time, os, numpy as np
from mpi4py import MPI
from model import ModelParallelStage
from dataset import load_parallel_io, load_broadcast
from caliper_wrapper import caliper_region

def train_model_parallel(args, comm, rank, size, timer):
    nt = int(os.environ.get("OMP_NUM_THREADS",1))
    torch.set_num_threads(nt)

    with caliper_region("data_io"):
        timer.start("io_load")
        if args.io_mode == "parallel":
            ds = load_parallel_io(args.data_file, rank, size)
        else:
            ds = load_broadcast(args.data_file, rank, size, comm)
        timer.stop("io_load")

    # Gather all data to rank 0 (it feeds the pipeline)
    all_X = comm.gather(ds.X.numpy(), root=0)
    all_y = comm.gather(ds.y.numpy(), root=0)
    if rank == 0:
        Xf = torch.tensor(np.vstack(all_X), dtype=torch.float32)
        yf = torch.tensor(np.vstack(all_y), dtype=torch.float32)

    nf = comm.bcast(ds.n_features, root=0)
    nt_out = comm.bcast(ds.n_targets, root=0)

    model = ModelParallelStage(nf, args.hidden_dim, nt_out, args.n_hidden, rank, size)
    crit = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    losses = []
    comp_t, comm_t = 0.0, 0.0
    comm_bytes = 0  # total bytes sent by this rank

    with caliper_region("mp_training"):
        for ep in range(args.epochs):
            if rank == 0:
                n = Xf.shape[0]; idx = torch.randperm(n)
                batches = [(Xf[idx[i:i+args.batch_size]], yf[idx[i:i+args.batch_size]])
                           for i in range(0, n-args.batch_size+1, args.batch_size)]
            nb = len(batches) if rank == 0 else 0
            nb = comm.bcast(nb, root=0)
            el, lc = 0, 0

            for b in range(nb):
                # FORWARD
                if rank == 0:
                    Xb, yb = batches[b]
                    t0=time.perf_counter(); Xb.requires_grad_(True); act=model(Xb); comp_t+=time.perf_counter()-t0
                    act_np = act.detach().numpy(); yb_np = yb.numpy()
                    t0=time.perf_counter(); comm.send(act_np, dest=1, tag=b); comm.send(yb_np, dest=size-1, tag=1000+b); comm_t+=time.perf_counter()-t0
                    comm_bytes += act_np.nbytes + yb_np.nbytes
                elif rank == size-1:
                    t0=time.perf_counter(); a_np=comm.recv(source=rank-1,tag=b); y_np=comm.recv(source=0,tag=1000+b); comm_t+=time.perf_counter()-t0
                    a_in=torch.tensor(a_np,dtype=torch.float32,requires_grad=True); yb=torch.tensor(y_np,dtype=torch.float32)
                    t0=time.perf_counter(); pred=model(a_in); loss=crit(pred,yb); el+=loss.item(); lc+=1; comp_t+=time.perf_counter()-t0
                else:
                    t0=time.perf_counter(); a_np=comm.recv(source=rank-1,tag=b); comm_t+=time.perf_counter()-t0
                    a_in=torch.tensor(a_np,dtype=torch.float32,requires_grad=True)
                    t0=time.perf_counter(); act=model(a_in); comp_t+=time.perf_counter()-t0
                    act_np = act.detach().numpy()
                    t0=time.perf_counter(); comm.send(act_np, dest=rank+1, tag=b); comm_t+=time.perf_counter()-t0
                    comm_bytes += act_np.nbytes

                # BACKWARD
                if rank == size-1:
                    t0=time.perf_counter(); opt.zero_grad(); loss.backward(); opt.step(); gb=a_in.grad.numpy(); comp_t+=time.perf_counter()-t0
                    t0=time.perf_counter(); comm.send(gb, dest=rank-1, tag=2000+b); comm_t+=time.perf_counter()-t0
                    comm_bytes += gb.nbytes
                elif rank == 0:
                    t0=time.perf_counter(); g_np=comm.recv(source=1,tag=2000+b); comm_t+=time.perf_counter()-t0
                    t0=time.perf_counter(); opt.zero_grad(); act.backward(torch.tensor(g_np,dtype=torch.float32)); opt.step(); comp_t+=time.perf_counter()-t0
                else:
                    t0=time.perf_counter(); g_np=comm.recv(source=rank+1,tag=2000+b); comm_t+=time.perf_counter()-t0
                    t0=time.perf_counter(); opt.zero_grad(); act.backward(torch.tensor(g_np,dtype=torch.float32)); opt.step(); gb=a_in.grad.numpy(); comp_t+=time.perf_counter()-t0
                    t0=time.perf_counter(); comm.send(gb, dest=rank-1, tag=2000+b); comm_t+=time.perf_counter()-t0
                    comm_bytes += gb.nbytes

            avg = (el/max(lc,1)) if rank==size-1 else 0.0
            avg = comm.bcast(avg, root=size-1)
            losses.append(avg)
            if rank == 0: print(f"  [ModelParallel] Epoch {ep+1}/{args.epochs} | MSE: {avg:.6f}")

    timer.record("compute", comp_t); timer.record("communication", comm_t)
    return {"epoch_losses": losses, "final_loss": losses[-1] if losses else None,
            "comm_bytes_mb": comm_bytes / (1024 * 1024)}
