import argparse
import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Dataset_Pro
from metrics import NMSELoss
from models.GPT4CP import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM4CP with jammer-aware masking")
    parser.add_argument('--train-r-path', type=str, default='./H_U_his_train.mat',
                        help='Path to historical channel .mat file')
    parser.add_argument('--train-t-path', type=str, default='./H_D_pre_train.mat',
                        help='Path to future channel .mat file')
    parser.add_argument('--save-path', type=str, default='Weights/U2D_LLM4CP.pth',
                        help='Checkpoint save path')
    parser.add_argument('--save-every', type=int, default=20,
                        help='If >0, save a checkpoint every N epochs to <save-path>.epoch{E}.pth')
    parser.add_argument('--save-last', action='store_true',
                        help='If set, save the final model at the end of training to <save-path>.last.pth')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Optional pretrained checkpoint path for initialization')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-step-size', type=int, default=150,
                        help='Step size (epochs) for StepLR; set <=0 to disable decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Decay factor for StepLR; set <=0 or >=1 to disable decay')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--jam-head-lr-mult', type=float, default=2.0,
                        help='LR multiplier for jam_head params relative to base lr')
    parser.add_argument('--device', type=str, default='cuda', help='torch device string')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU index used for model instantiation (single GPU)')
    parser.add_argument('--multi-gpu', action='store_true', help='Enable torch.nn.DataParallel over multiple GPUs')
    parser.add_argument('--device-ids', type=str, default='0',
                        help='Comma separated GPU ids used when --multi-gpu is enabled')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--few-shot', action='store_true', help='Enable few-shot sampling (keep every 10th sample)')
    parser.add_argument('--use-jammer', action='store_true', help='Enable synthetic jammer augmentation and mask loss')
    parser.add_argument('--jammer-cfg', type=str, default=None,
                        help='Path to JSON config overriding jammer parameters')
    parser.add_argument('--lambda-mask', type=float, default=0, help='Initial weight for jammer mask BCE loss')
    parser.add_argument('--lambda-mask-final', type=float, default=None,
                        help='Optional final weight for jammer mask BCE loss after scheduling')
    parser.add_argument('--lambda-mask-switch', type=int, default=0,
                        help='Epoch index to start transitioning lambda-mask (0-based)')
    parser.add_argument('--lambda-mask-ramp', type=int, default=0,
                        help='Number of epochs to linearly ramp lambda-mask from start to final after switch')
    parser.add_argument('--jam-gate-strength', type=float, default=0,
                        help='Target multiplier applied to predicted jam mask when gating inputs')
    parser.add_argument('--jam-gate-start', type=float, default=0.0,
                        help='Initial jam gate strength before warmup ramp')
    parser.add_argument('--jam-gate-warmup', type=int, default=100,
                        help='Epochs to linearly ramp jam gate from start to target (0 to disable)')
    parser.add_argument('--jam-head-hidden-ratio', type=float, default=0.75,
                        help='Hidden size ratio for jam head MLP (0-1)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='If set, append per-epoch summaries to this file')
    parser.add_argument('--eval-clean', action='store_true',
                        help='Run an additional validation pass without jammers (clean)')
    parser.add_argument('--is-u2d', action='store_true',
                        help='Use U->D (FDD) target: read H_D_pre_* keys instead of H_U_pre_* in Dataset_Pro')
    return parser.parse_args()


def parse_device_ids(device_ids: str) -> List[int]:
    ids = []
    for token in device_ids.split(','):
        token = token.strip()
        if token:
            ids.append(int(token))
    return ids


def load_pretrained(model: Model, path: Optional[str], device: torch.device):
    if not path or not os.path.exists(path):
        return
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, Model):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, nn.DataParallel):
        state_dict = checkpoint.module.state_dict()
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    # Filter out keys with shape mismatch (e.g., jam_head when hidden ratio changed)
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and isinstance(v, torch.Tensor) and isinstance(model_sd[k], torch.Tensor):
            if v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            # keep non-tensor buffers or allow missing keys
            filtered[k] = v
    if skipped:
        print("[Init] Skipped mismatched parameters:")
        for name, s_old, s_new in skipped:
            print(f"        {name}: ckpt {s_old} != model {s_new}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[Init] Missing parameters: {missing}")
    if unexpected:
        print(f"[Init] Unexpected parameters: {unexpected}")


def save_best_checkpoint(model: Model, path: str):
    torch.save(model, path)


def _core_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_loop(training_loader, validation_loader, model, optimizer, criterion_pred,
               criterion_mask, args, device, scheduler=None, gate_schedule=None,
               validation_loader_clean=None, lambda_schedule=None):
    best_loss = float('inf')
    print('Start training...')
    log_fp = None
    if args.log_file:
        try:
            log_fp = open(args.log_file, 'a')
        except Exception as e:
            print(f"[Log] Failed to open log file {args.log_file}: {e}")
    for epoch in range(args.epochs):
        if gate_schedule is not None and args.use_jammer:
            gate_value = gate_schedule(epoch)
            core = _core_model(model)
            if hasattr(core, 'use_jam_head') and core.use_jam_head:
                core.jam_gate_strength = gate_value
        else:
            gate_value = getattr(_core_model(model), 'jam_gate_strength', None)

        if lambda_schedule is not None and args.use_jammer:
            lambda_mask_weight = lambda_schedule(epoch)
        else:
            lambda_mask_weight = args.lambda_mask
        epoch_train_loss, epoch_train_mask = [], []
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        for iteration, batch in enumerate(training_loader, 1):
            if args.use_jammer:
                pred_t, prev, jam_mask = batch
                jam_mask = jam_mask.to(device)
                return_mask = True
            else:
                pred_t, prev = batch
                jam_mask = None
                return_mask = False
            pred_t = pred_t.to(device)
            prev = prev.to(device)
            optimizer.zero_grad()
            outputs = model(prev, None, None, None, return_mask=return_mask)
            if args.use_jammer:
                pred_m, pred_mask = outputs
            else:
                pred_m = outputs
                pred_mask = None

            loss = criterion_pred(pred_m, pred_t)
            total_loss = loss
            if args.use_jammer and pred_mask is not None and jam_mask is not None:
                mask_loss = criterion_mask(pred_mask, jam_mask)
                total_loss = total_loss + lambda_mask_weight * mask_loss
                epoch_train_mask.append(mask_loss.item())
            total_loss.backward()
            optimizer.step()
            epoch_train_loss.append(total_loss.item())

        mean_train = np.nanmean(np.array(epoch_train_loss)) if epoch_train_loss else float('nan')
        mean_mask = np.nanmean(np.array(epoch_train_mask)) if epoch_train_mask else 0.0
        if args.use_jammer:
            gate_info = f" gate: {gate_value:.4f}" if gate_value is not None else ""
            lambda_info = f" lambda: {lambda_mask_weight:.4f}"
            print(f"Epoch: {epoch + 1}/{args.epochs} training loss: {mean_train:.7f} "
                  f"mask: {mean_mask:.7f} lr: {current_lr:.6e}{gate_info}{lambda_info}")
        else:
            print(f"Epoch: {epoch + 1}/{args.epochs} training loss: {mean_train:.7f} "
                  f"lr: {current_lr:.6e}")

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_mask, epoch_val_nmse = [], [], []
        with torch.no_grad():
            for iteration, batch in enumerate(validation_loader, 1):
                if args.use_jammer:
                    pred_t, prev, jam_mask = batch
                    jam_mask = jam_mask.to(device)
                    return_mask = True
                else:
                    pred_t, prev = batch
                    jam_mask = None
                    return_mask = False
                pred_t = pred_t.to(device)
                prev = prev.to(device)
                outputs = model(prev, None, None, None, return_mask=return_mask)
                if args.use_jammer:
                    pred_m, pred_mask = outputs
                else:
                    pred_m = outputs
                    pred_mask = None
                nmse_loss = criterion_pred(pred_m, pred_t)
                total_loss = nmse_loss
                if args.use_jammer and pred_mask is not None and jam_mask is not None:
                    mask_loss = criterion_mask(pred_mask, jam_mask)
                    total_loss = total_loss + lambda_mask_weight * mask_loss
                    epoch_val_mask.append(mask_loss.item())
                epoch_val_loss.append(total_loss.item())
                epoch_val_nmse.append(nmse_loss.item())

        v_loss = np.nanmean(np.array(epoch_val_loss)) if epoch_val_loss else float('nan')
        v_nmse = np.nanmean(np.array(epoch_val_nmse)) if epoch_val_nmse else float('nan')
        if args.use_jammer and epoch_val_mask:
            v_mask = np.nanmean(np.array(epoch_val_mask))
            print(f"Validate loss: {v_loss:.7f} (nmse: {v_nmse:.7f}) mask: {v_mask:.7f}")
        else:
            print(f"Validate loss: {v_loss:.7f} (nmse: {v_nmse:.7f})")

        # Optional clean validation (no jammers)
        v_clean = None
        if validation_loader_clean is not None:
            epoch_val_clean = []
            with torch.no_grad():
                for iteration, batch in enumerate(validation_loader_clean, 1):
                    pred_t, prev = batch
                    pred_t = pred_t.to(device)
                    prev = prev.to(device)
                    pred_m = model(prev, None, None, None, return_mask=False)
                    loss_clean = criterion_pred(pred_m, pred_t)
                    epoch_val_clean.append(loss_clean.item())
            v_clean = np.nanmean(np.array(epoch_val_clean)) if epoch_val_clean else float('nan')
            print(f"Validate(clean) nmse: {v_clean:.7f}")
        if v_loss < best_loss:
            best_loss = v_loss
            save_best_checkpoint(model, args.save_path)
            # also optionally write a numbered best file for easier bookkeeping
            try:
                best_epoch_path = args.save_path + f".best_epoch{epoch+1}.pth"
                torch.save(_core_model(model).state_dict(), best_epoch_path)
            except Exception:
                # non-fatal: ignore failures to write the auxiliary file
                pass
        # periodic save if requested
        if getattr(args, 'save_every', 0) and (epoch + 1) % int(args.save_every) == 0:
            try:
                periodic_path = args.save_path + f".epoch{epoch+1}.pth"
                torch.save(model, periodic_path)
                print(f"[Save] Periodic checkpoint saved to {periodic_path}")
            except Exception as e:
                print(f"[Save] Failed to write periodic checkpoint: {e}")
        if scheduler is not None:
            scheduler.step()
            next_lr = optimizer.param_groups[0]['lr']
            if next_lr != current_lr:
                print(f"[LR] Updated learning rate to {next_lr:.6e}")

        # Logging to file
        if log_fp is not None:
            try:
                if args.use_jammer and epoch_val_mask:
                    v_mask = np.nanmean(np.array(epoch_val_mask))
                else:
                    v_mask = float('nan')
                log_fp.write(
                    f"epoch={epoch+1}, lr={current_lr:.6e}, train_total={mean_train:.7f}, "
                    f"train_mask={mean_mask:.7f}, val_total={v_loss:.7f}, val_nmse={v_nmse:.7f}, "
                    f"val_mask={v_mask:.7f}, val_clean={v_clean if v_clean is not None else float('nan'):.7f}, "
                    f"gate={gate_value if gate_value is not None else float('nan'):.6f}, "
                    f"lambda_mask={lambda_mask_weight:.6f}\n"
                )
                log_fp.flush()
            except Exception as e:
                print(f"[Log] Write failed: {e}")

    # end of epoch loop
    # after training completes, optionally save the final checkpoint
    if getattr(args, 'save_last', False):
        try:
            final_path = args.save_path + ".last.pth"
            torch.save(model, final_path)
            print(f"[Save] Final checkpoint saved to {final_path}")
        except Exception as e:
            print(f"[Save] Failed to write final checkpoint: {e}")

    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass


def main():
    args = parse_args()
    if args.multi_gpu:
        device_ids = parse_device_ids(args.device_ids)
        if not device_ids:
            raise ValueError("--multi-gpu enabled but no GPU ids provided")
        torch.cuda.set_device(device_ids[0])
        primary_device = torch.device(f'cuda:{device_ids[0]}')
        primary_gpu = device_ids[0]
    else:
        device_ids = None
        primary_device = torch.device(args.device)
        primary_gpu = args.gpu_id

    jammer_cfg = None
    if args.jammer_cfg:
        with open(args.jammer_cfg, 'r') as f:
            jammer_cfg = json.load(f)

    train_set = Dataset_Pro(args.train_r_path, args.train_t_path, is_train=1, is_U2D=1 if args.is_u2d else 0,
                            is_few=1 if args.few_shot else 0,
                            use_jammer=args.use_jammer, jammer_cfg=jammer_cfg,
                            return_mask=args.use_jammer)
    validate_set = Dataset_Pro(args.train_r_path, args.train_t_path, is_train=0, is_U2D=1 if args.is_u2d else 0,
                               is_few=0,
                               use_jammer=args.use_jammer, jammer_cfg=jammer_cfg,
                               return_mask=args.use_jammer)
    validate_set_clean = None
    validation_loader_clean = None
    if args.eval_clean and args.use_jammer:
        validate_set_clean = Dataset_Pro(args.train_r_path, args.train_t_path, is_train=0, is_U2D=1 if args.is_u2d else 0,
                                         is_few=0,
                                         use_jammer=False, jammer_cfg=None,
                                         return_mask=False)

    model = Model(gpu_id=primary_gpu,
                  pred_len=4, prev_len=16,
                  UQh=1, UQv=1, BQh=1, BQv=1,
                  use_jam_head=args.use_jammer,
                  jam_gate_strength=args.jam_gate_start if args.use_jammer else args.jam_gate_strength,
                  jam_head_hidden_ratio=args.jam_head_hidden_ratio).to(primary_device)
    load_pretrained(model, args.pretrained_path, primary_device)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=device_ids).to(primary_device)

    total = sum(param.nelement() for param in model.parameters())
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    training_loader = DataLoader(
        dataset=train_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        dataset=validate_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.eval_clean and args.use_jammer and (validate_set_clean is not None):
        validation_loader_clean = DataLoader(
            dataset=validate_set_clean,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    # Build optimizer with param groups: jam_head gets higher LR if requested
    core = _core_model(model)
    if args.use_jammer and hasattr(core, 'jam_head') and core.jam_head is not None:
        jam_head_module = core.jam_head
        if isinstance(jam_head_module, nn.Module):
            jam_params = list(jam_head_module.parameters())
        else:
            jam_params = []
        if jam_params:
            jam_param_ids = {id(p) for p in jam_params}
            base_params = [p for p in model.parameters() if id(p) not in jam_param_ids]
            optimizer = optim.Adam(
                [
                    {"params": base_params, "lr": args.lr, "betas": (0.9, 0.999), "weight_decay": args.weight_decay},
                    {"params": jam_params, "lr": args.lr * max(1.0, args.jam_head_lr_mult), "betas": (0.9, 0.999), "weight_decay": args.weight_decay},
                ]
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                   weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                               weight_decay=args.weight_decay)
    criterion_pred = NMSELoss().to(primary_device)
    criterion_mask = torch.nn.BCELoss().to(primary_device) if args.use_jammer else None

    scheduler = None
    if args.lr_step_size > 0 and 0 < args.lr_gamma < 1:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

    gate_schedule_fn = None
    if args.use_jammer:
        gate_start = max(0.0, args.jam_gate_start)
        gate_target = max(0.0, args.jam_gate_strength)
        warmup_epochs = max(0, args.jam_gate_warmup)

        if warmup_epochs <= 0 or abs(gate_target - gate_start) < 1e-8:
            def _constant_gate(epoch: int, _gate=gate_target):
                return _gate

            gate_schedule_fn = _constant_gate
        else:
            def _warmup_gate(epoch: int, _start=gate_start, _target=gate_target, _warm=warmup_epochs):
                progress = min(max(epoch / float(_warm), 0.0), 1.0)
                return _start + (_target - _start) * progress

            gate_schedule_fn = _warmup_gate

    lambda_schedule_fn = None
    if args.use_jammer and args.lambda_mask_final is not None:
        lambda_start = args.lambda_mask
        lambda_final = args.lambda_mask_final
        if abs(lambda_final - lambda_start) < 1e-8:
            lambda_schedule_fn = None
        else:
            switch_epoch = max(0, args.lambda_mask_switch)
            ramp_epochs = max(0, args.lambda_mask_ramp)

            if ramp_epochs <= 0:
                def _lambda_step(epoch: int, _start=lambda_start, _final=lambda_final, _switch=switch_epoch):
                    return _start if epoch < _switch else _final

                lambda_schedule_fn = _lambda_step
            else:
                def _lambda_ramp(epoch: int, _start=lambda_start, _final=lambda_final,
                                 _switch=switch_epoch, _ramp=ramp_epochs):
                    if epoch < _switch:
                        return _start
                    progress = min(max((epoch - _switch) / float(_ramp), 0.0), 1.0)
                    return _start + (_final - _start) * progress

                lambda_schedule_fn = _lambda_ramp

    train_loop(
        training_loader,
        validation_loader,
        model,
        optimizer,
        criterion_pred,
        criterion_mask,
        args,
        primary_device,
        scheduler=scheduler,
        gate_schedule=gate_schedule_fn,
        validation_loader_clean=validation_loader_clean,
        lambda_schedule=lambda_schedule_fn,
    )

    total = sum(param.nelement() for param in model.parameters())
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))


if __name__ == "__main__":
    main()
