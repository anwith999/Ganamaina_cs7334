"""Sequential baseline — single process, no parallelism."""
import torch, torch.nn as nn, os
from model import RegressionNN
from dataset import HDF5RegressionDataset, make_loader
from caliper_wrapper import caliper_region

def train_sequential(args, timer):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS",1)))
    ds = HDF5RegressionDataset(args.data_file)
    loader = make_loader(ds, args.batch_size)
    model = RegressionNN(ds.n_features, args.hidden_dim, ds.n_targets, args.n_hidden)
    crit = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    losses = []
    timer.start("compute")
    with caliper_region("sequential_training"):
        for ep in range(args.epochs):
            el, nb = 0, 0
            for Xb, yb in loader:
                opt.zero_grad(); pred = model(Xb); loss = crit(pred, yb)
                loss.backward(); opt.step(); el += loss.item(); nb += 1
            avg = el/nb; losses.append(avg)
            print(f"  [Sequential] Epoch {ep+1}/{args.epochs} | MSE: {avg:.6f}")
    timer.stop("compute"); timer.record("communication", 0)
    return {"epoch_losses": losses, "final_loss": losses[-1], "comm_bytes_mb": 0.0}
