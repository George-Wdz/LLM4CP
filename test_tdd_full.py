"""
@Project ：LLM4CP
@File    ：test.py
@IDE     ：PyCharm
@Author  ：XvanyvLiu
@mail    : xvanyvliu@gmail.com
@Date    ：2024/4/8 17:11
"""
import time
import argparse
import json
import torch
import numpy as np
from data import LoadBatch_ofdm_1, LoadBatch_ofdm_2, noise, Transform_TDD_FDD, apply_jammers
from metrics import NMSELoss, SE_Loss
from einops import rearrange
import hdf5storage
import tqdm
from pvec import pronyvec
from PAD import PAD3

def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--prev-path', default="../Dataset/test/H_U_his_test.mat")
    ap.add_argument('--pred-path', default="../Dataset/test/H_U_pre_test.mat")
    ap.add_argument('--pred-path-fdd', default="../Dataset/test/H_D_pre_test.mat")
    ap.add_argument('--is-u2d', action='store_true', help='Use U2D FDD target for evaluation')
    # models / device
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--models', nargs='*', default=['gpt', 'transformer', 'cnn', 'gru', 'lstm', 'rnn', 'np', 'pad'])
    ap.add_argument('--weights-gpt', default='../Weights/full_shot_tdd/U2U3.5_LLM4CP_tdd_light_sched.pth')
    ap.add_argument('--weights-transformer', default='../Weights/full_shot_tdd/U2U_trans.pth')
    ap.add_argument('--weights-cnn', default='../Weights/full_shot_tdd/U2U_cnn.pth')
    ap.add_argument('--weights-gru', default='../Weights/full_shot_tdd/U2U_gru.pth')
    ap.add_argument('--weights-lstm', default='../Weights/full_shot_tdd/U2U_lstm.pth')
    ap.add_argument('--weights-rnn', default='../Weights/full_shot_tdd/U2U_rnn.pth')
    # evaluation settings
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--snr-awgn', type=float, default=18.0, help='AWGN SNR used for test-time noise() on both input/target')
    ap.add_argument('--prev-len', type=int, default=16)
    ap.add_argument('--label-len', type=int, default=12)
    ap.add_argument('--pred-len', type=int, default=4)
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--Nt', type=int, default=16)
    ap.add_argument('--Nr', type=int, default=1)
    # jammer injection (inputs only)
    ap.add_argument('--use-jammer', action='store_true')
    ap.add_argument('--jammer-cfg', default=None, help='Path to jammer JSON config')
    ap.add_argument('--jam-gate', type=float, default=None, help='If set and model is GPT4CP with jam head, override jam_gate_strength during inference')
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    is_U2D = 1 if args.is_u2d else 0
    prev_path = args.prev_path
    pred_path = args.pred_path
    pred_path_fdd = args.pred_path_fdd
    model_path = {
        'gpt': args.weights_gpt,
        'transformer': args.weights_transformer,
        'cnn': args.weights_cnn,
        'gru': args.weights_gru,
        'lstm': args.weights_lstm,
        'rnn': args.weights_rnn
    }
    model_test_enable = args.models
    prev_len = args.prev_len
    label_len = args.label_len
    pred_len = args.pred_len
    K, Nt, Nr, SR = (args.K, args.Nt, args.Nr, 1)
    print("Total model nums:", len(model_test_enable))
    # load model and test
    criterion = NMSELoss()
    criterion_se = SE_Loss(snr=10, device=device)
    NMSE = [[] for i in model_test_enable]
    SE = [[] for i in model_test_enable]
    test_data_prev_base = hdf5storage.loadmat(prev_path)['H_U_his_test']
    if is_U2D:
        test_data_pred_base = hdf5storage.loadmat(pred_path_fdd)['H_D_pre_test']
    else:
        test_data_pred_base = hdf5storage.loadmat(pred_path)['H_U_pre_test']
    # load jammer cfg once if provided
    jammer_cfg = None
    if args.use_jammer:
        if args.jammer_cfg is not None:
            with open(args.jammer_cfg, 'r') as f:
                jammer_cfg = json.load(f)
        else:
            jammer_cfg = {}
    for i in range(len(model_test_enable)):
        print("---------------------------------------------------------------")
        print("loading ", i + 1, "th model......", model_test_enable[i])
        model = None
        if model_test_enable[i] not in ['pad', 'pvec', 'np']:
            model = torch.load(model_path[model_test_enable[i]], map_location=device).to(device)
            # optionally override jam gate for GPT-like model
            if model_test_enable[i] == 'gpt' and args.jam_gate is not None and hasattr(model, 'jam_gate_strength'):
                try:
                    model.jam_gate_strength = float(args.jam_gate)
                    print(f"[Info] Set jam_gate_strength = {model.jam_gate_strength}")
                except Exception as e:
                    print(f"[Warn] Failed to set jam_gate_strength: {e}")
        for speed in range(0, 10):
            test_loss_stack = []
            test_loss_stack_se = []
            test_loss_stack_se0 = []
            test_data_prev = test_data_prev_base[[speed], ...]
            test_data_pred = test_data_pred_base[[speed], ...]
            test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')
            # add AWGN to both inputs and targets (kept for legacy fairness)
            test_data_prev = noise(test_data_prev, args.snr_awgn)
            test_data_pred = noise(test_data_pred, args.snr_awgn)
            # optionally apply jammers to inputs ONLY, before normalization
            if args.use_jammer:
                # reshape (B, M, L, K) -> (B, L, M*K) to use apply_jammers with K known
                Bm, M, Ldim, Kdim = test_data_prev.shape
                if Kdim != K:
                    print(f"[Warn] Jammer K({K}) != data K({Kdim}); using data K={Kdim} for jammers.")
                K_eff = Kdim
                prev_reshaped = np.reshape(np.transpose(test_data_prev, (0, 2, 1, 3)), (Bm, Ldim, M * K_eff))
                # patch cfg with effective subcarrier count
                cfg_eff = dict(jammer_cfg or {})
                cfg_eff['num_subcarriers'] = int(K_eff)
                prev_jammed, _ = apply_jammers(prev_reshaped, cfg_eff)
                # back to original layout (B, M, L, K)
                test_data_prev = np.transpose(np.reshape(prev_jammed, (Bm, Ldim, M, K_eff)), (0, 2, 1, 3))
            std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
            test_data_prev = test_data_prev / std
            test_data_pred = test_data_pred / std
            lens, _, _, _ = test_data_prev.shape
            if model_test_enable[i] in ['gpt', 'transformer', 'rnn', 'lstm', 'gru', 'cnn', 'np']:
                if model is not None:
                    model.eval()
                prev_data = LoadBatch_ofdm_2(test_data_prev)
                pred_data = LoadBatch_ofdm_2(test_data_pred)
                bs = args.batch_size
                cycle_times = lens // bs
                with torch.no_grad():
                    for cyt in range(cycle_times):
                        prev = prev_data[cyt * bs:(cyt + 1) * bs, :, :].to(device)
                        pred = pred_data[cyt * bs:(cyt + 1) * bs, :, :].to(device)
                        prev = rearrange(prev, 'b m l k -> (b m) l k')
                        pred = rearrange(pred, 'b m l k -> (b m) l k')
                        # default np baseline behavior to ensure 'out' always defined
                        out = prev[:, [-1], :].repeat([1, pred_len, 1])
                        if model_test_enable[i] == 'gpt':
                            if model is None:
                                raise RuntimeError("GPT model weights not loaded.")
                            out = model(prev, None, None, None)
                        elif model_test_enable[i] == 'transformer':
                            encoder_input = prev
                            dec_inp = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
                            decoder_input = torch.cat([encoder_input[:, prev_len - label_len:prev_len, :], dec_inp],
                                                      dim=1)
                            if model is not None:
                                out = model(encoder_input, decoder_input)
                        elif model_test_enable[i] in ['lstm', 'rnn', 'gru']:
                            if model is not None:
                                out = model(prev, pred_len, device)
                        elif model_test_enable[i] == 'cnn':
                            if model is not None:
                                out = model(prev)
                        elif model_test_enable[i] == 'np':
                            out = prev[:, [-1], :].repeat([1, pred_len, 1])
                        loss = criterion(out, pred)
                        out = rearrange(out, '(b m) l k -> b l (k m)', b=bs)
                        pred = rearrange(pred, '(b m) l k -> b l (k m)', b=bs)
                        se, se0 = criterion_se(h=Transform_TDD_FDD(out, Nt=4*4, Nr=1),
                                               h0=Transform_TDD_FDD(pred, Nt=4*4, Nr=1))
                        test_loss_stack.append(loss.item())
                        test_loss_stack_se.append(se.item())
                        test_loss_stack_se0.append(se0.item())
                print("speed", speed, ":  NMSE:", np.nanmean(np.array(test_loss_stack)),
                      "SE:", -np.nanmean(np.array(test_loss_stack_se)), "SE0:", -np.nanmean(np.array(test_loss_stack_se0)),
                      "SE_per", np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
                NMSE[i].append(np.nanmean(np.array(test_loss_stack)))
                SE[i].append(np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
            elif model_test_enable[i] in ['pad', 'pvec']:
                cycle_times = lens
                for cyt in range(cycle_times):
                    prev = test_data_prev[cyt, :, :, :]
                    prev = rearrange(prev, 'm l k -> k l m', k=K)
                    pred = test_data_pred[cyt, :, :, :]
                    pred = rearrange(pred, 'm l k -> k l m', k=K)
                    if model_test_enable[i] == 'pad':
                        # outputs_AR_delay
                        out = PAD3(prev, p=8, startidx=prev_len, subcarriernum=K, Nr=Nr, Nt=Nt,
                                   pre_len=pred_len)
                    elif model_test_enable[i] == 'pvec':
                        # outputs_AR_freq
                        out = pronyvec(prev, p=8, startidx=prev_len, subcarriernum=K, Nr=Nr, Nt=Nt,
                                       pre_len=pred_len)
                    out = LoadBatch_ofdm_1(out)
                    pred = LoadBatch_ofdm_1(pred)
                    loss = criterion(out, pred)
                    se, se0 = criterion_se(h=Transform_TDD_FDD(out, Nt=4*4, Nr=1), h0=Transform_TDD_FDD(pred, Nt=4*4, Nr=1))
                    test_loss_stack.append(loss.item())
                    test_loss_stack_se.append(se.item())
                    test_loss_stack_se0.append(se0.item())
                print("speed", speed, ":  NMSE:", np.nanmean(np.array(test_loss_stack)),
                      "SE:", -np.nanmean(np.array(test_loss_stack_se)), "SE0:", -np.nanmean(np.array(test_loss_stack_se0)),
                      "SE_per", np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))
                NMSE[i].append(np.nanmean(np.array(test_loss_stack)))
                SE[i].append(np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0)))

    fout_nmse = open(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + "_data_nmse_tdd_full.csv", "w")
    for row in NMSE:
        row = list(map(str, row))
        fout_nmse.write(','.join(row))
        fout_nmse.write('\n')
    fout_nmse.close()
