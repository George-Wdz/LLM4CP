import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

from data import Dataset_Pro
from metrics import NMSELoss
from models.model import RNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN baseline on LLM4CP data")
    # data paths
    parser.add_argument('--train-r-path', type=str, required=True,
                        help='Path to historical channel .mat file (e.g., Dataset/train_data/H_U_his_train.mat)')
    parser.add_argument('--train-t-path', type=str, required=True,
                        help='Path to future channel .mat file (e.g., Dataset/train_data/H_U_pre_train.mat)')
    parser.add_argument('--save-path', type=str, required=True,
                        help='File to store the best checkpoint (saved with torch.save(model, path))')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Optional RNN checkpoint to resume from (expects torch.save(model, path) format)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Optional log file; writes CSV with epoch metrics')
    parser.add_argument('--few-shot', action='store_true',
                        help='Enable few-shot regime (keep every 10th sample)')
    parser.add_argument('--is-u2d', action='store_true',
                        help='Switch to FDD setting (read H_D_pre_* targets)')
    parser.add_argument('--use-jammer', action='store_true',
                        help='Augment history with synthetic jammers during training')
    parser.add_argument('--jammer-cfg', type=str, default=None,
                        help='Path to jammer JSON config (only when --use-jammer)')
    parser.add_argument('--no-awgn', action='store_true',
                        help='Disable AWGN injection inside Dataset_Pro')

    # optimization
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='Optional validation batch size; defaults to --batch-size when unset')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lr-step-size', type=int, default=100,
                        help='Epoch interval for StepLR (<=0 disables scheduler)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor for StepLR when enabled')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='Gradient clipping value; <=0 disables clipping')

    # architecture
    parser.add_argument('--input-size', type=int, default=None,
                        help='Linear embedding size before recurrent core (defaults to feature dimension)')
    parser.add_argument('--hidden-size', type=int, default=192)
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of stacked RNN layers (baseline uses 4)')

    # runtime
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-parallel', action='store_true',
                        help='Enable DataParallel over all visible CUDA devices')

    return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_state_dict(path: Optional[str], device: torch.device) -> Optional[dict]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    print(f"[Resume] Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, nn.DataParallel):
        return checkpoint.module.state_dict()
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            return checkpoint['state_dict']
        return checkpoint
    raise ValueError(f"Unsupported checkpoint type at {path}: {type(checkpoint)}")


def _save_model(model: nn.Module, path: str, device: torch.device) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    model_cpu = base_model.to('cpu')
    torch.save(model_cpu, path)
    base_model.to(device)
    if isinstance(model, nn.DataParallel):
        model.to(device)
    print(f"[Save] Best checkpoint updated -> {path}")


def prepare_dataloaders(args: argparse.Namespace):
    jammer_cfg = None
    if args.use_jammer and args.jammer_cfg:
        with open(args.jammer_cfg, 'r') as cfg_fp:
            jammer_cfg = json.load(cfg_fp)

    train_set = Dataset_Pro(
        args.train_r_path,
        args.train_t_path,
        is_train=1,
        is_U2D=1 if args.is_u2d else 0,
        is_few=1 if args.few_shot else 0,
        use_jammer=args.use_jammer,
        jammer_cfg=jammer_cfg,
        return_mask=False,
        add_awgn=not args.no_awgn,
    )
    val_set = Dataset_Pro(
        args.train_r_path,
        args.train_t_path,
        is_train=0,
        is_U2D=1 if args.is_u2d else 0,
        is_few=0,
        use_jammer=args.use_jammer,
        jammer_cfg=jammer_cfg,
        return_mask=False,
        add_awgn=not args.no_awgn,
    )

    loader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_bs = args.val_batch_size or args.batch_size
    loader_val = DataLoader(
        val_set,
        batch_size=val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    return train_set, loader_train, loader_val


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, *, device: torch.device,
              pred_len: int, optimizer: Optional[optim.Optimizer] = None, grad_clip: float = 0.0) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0.0
    sample_count = 0
    # when DataParallel is active we let the module infer device from inputs
    forward_device = None if isinstance(model, nn.DataParallel) else device

    for gt_pred, hist in loader:
        hist = hist.to(device)
        gt_pred = gt_pred.to(device)
        out = model(hist, pred_len, forward_device)
        loss = criterion(out, gt_pred)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = gt_pred.size(0)
        epoch_loss += loss.item() * batch_size
        sample_count += batch_size

    return epoch_loss / max(sample_count, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_set, loader_train, loader_val = prepare_dataloaders(args)
    prev_len = train_set.prev.shape[1]
    pred_len = train_set.pred.shape[1]
    feature_dim = train_set.prev.shape[2]
    if prev_len != 16 or pred_len != 4:
        raise ValueError(f"RNN baseline expects prev_len=16 and pred_len=4, but got prev_len={prev_len}, pred_len={pred_len}")

    input_size = args.input_size or feature_dim
    model = RNN(features=feature_dim, input_size=input_size, hidden_size=args.hidden_size, num_layers=args.layers)
    model = model.to(device)

    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"[Info] Enabling DataParallel across {torch.cuda.device_count()} devices")
        model = nn.DataParallel(model)

    state_dict = _load_state_dict(args.pretrained_path, device)
    if state_dict is not None:
        target = model.module if isinstance(model, nn.DataParallel) else model
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Resume] Missing parameters: {missing}")
        if unexpected:
            print(f"[Resume] Unexpected parameters: {unexpected}")

    criterion = NMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_step_size and args.lr_step_size > 0 and 0 < args.lr_gamma < 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    best_val = float('inf')
    log_fp = None
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fp = open(args.log_file, 'a')

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                model,
                loader_train,
                criterion,
                device=device,
                pred_len=pred_len,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
            )

            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    loader_val,
                    criterion,
                    device=device,
                    pred_len=pred_len,
                    optimizer=None,
                    grad_clip=0.0,
                )

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | train_nmse={train_loss:.6f} | val_nmse={val_loss:.6f} | lr={current_lr:.6e}")

            if log_fp is not None:
                log_fp.write(
                    f"epoch={epoch}, lr={current_lr:.6e}, train_nmse={train_loss:.6f}, val_nmse={val_loss:.6f}\n"
                )
                log_fp.flush()

            if val_loss < best_val:
                best_val = val_loss
                _save_model(model, args.save_path, device)
    finally:
        if log_fp is not None:
            log_fp.close()

    print(f"Training complete. Best validation NMSE: {best_val:.6f}")


if __name__ == '__main__':
    main()
