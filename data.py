import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from einops import rearrange
from numpy import random


DEFAULT_JAMMER_CFG = {
    'types': ['wb', 'pb', 'tone', 'pulse', 'fh'],
    'min_types': 1,
    'max_types': 2,
    'jam_prob': 1.0,
    'jsr_db_range': (-10.0, 20.0),
    'pb_bandwidth_frac_range': (0.1, 0.4),
    'tone_count_range': (2, 6),
    'pulse_duty_cycle_range': (0.05, 0.2),
    'fh_hop_len_range': (1, 4),
    'fh_bandwidth_frac_range': (0.08, 0.2),
    'num_subcarriers': 48,
    'seed': None
}


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise


def _merge_jammer_cfg(user_cfg):
    cfg = DEFAULT_JAMMER_CFG.copy()
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _ensure_rng(seed):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def _compute_jam_sigma(sample, jsr_db):
    signal_power = np.mean(np.abs(sample) ** 2) + 1e-12
    return np.sqrt(signal_power * (10.0 ** (jsr_db / 10.0)) / 2.0)


def _select_band(rng, total_bins, frac_range):
    width = max(1, int(round(total_bins * rng.uniform(*frac_range))))
    width = min(width, total_bins)
    start = rng.integers(0, total_bins)
    indices = (np.arange(width) + start) % total_bins
    return indices


def _apply_wb(sample, mask, jsr_db, rng):
    sigma = _compute_jam_sigma(sample, jsr_db)
    jam = sigma * (rng.standard_normal(sample.shape) + 1j * rng.standard_normal(sample.shape))
    sample = sample + jam
    mask[:] = np.maximum(mask, 1.0)
    return sample, mask


def _apply_pb(sample, mask, jsr_db, rng, K, group, cfg):
    sigma = _compute_jam_sigma(sample, jsr_db)
    sample_view = sample.reshape(sample.shape[0], K, group)
    mask_view = mask.reshape(mask.shape[0], K, group)
    indices = _select_band(rng, K, cfg['pb_bandwidth_frac_range'])
    jam = sigma * (rng.standard_normal(sample_view[:, indices, :].shape) +
                   1j * rng.standard_normal(sample_view[:, indices, :].shape))
    sample_view[:, indices, :] = sample_view[:, indices, :] + jam
    mask_view[:, indices, :] = np.maximum(mask_view[:, indices, :], 1.0)
    return sample_view.reshape(sample.shape), mask_view.reshape(mask.shape)


def _apply_tone(sample, mask, jsr_db, rng, K, group, cfg):
    sigma = _compute_jam_sigma(sample, jsr_db)
    sample_view = sample.reshape(sample.shape[0], K, group)
    mask_view = mask.reshape(mask.shape[0], K, group)
    tone_low, tone_high = cfg['tone_count_range']
    tone_count = int(rng.integers(max(1, tone_low), max(2, tone_high + 1)))
    tone_indices = rng.choice(K, size=min(tone_count, K), replace=False)
    t = np.arange(sample.shape[0])[None, :, None]
    phases = rng.uniform(0.0, 2 * np.pi, size=(tone_indices.shape[0], 1, group))
    omegas = rng.uniform(0.0, np.pi, size=(tone_indices.shape[0], 1, 1))
    tone = sigma * np.exp(1j * (omegas * t + phases))
    for tone_idx, freq_idx in enumerate(tone_indices):
        tone_wave = tone[tone_idx]
        sample_view[:, freq_idx, :] = sample_view[:, freq_idx, :] + tone_wave
        mask_view[:, freq_idx, :] = np.maximum(mask_view[:, freq_idx, :], 1.0)
    return sample_view.reshape(sample.shape), mask_view.reshape(mask.shape)


def _apply_pulse(sample, mask, jsr_db, rng, cfg):
    sigma = _compute_jam_sigma(sample, jsr_db)
    T = sample.shape[0]
    duty_low, duty_high = cfg['pulse_duty_cycle_range']
    duty_cycle = np.clip(rng.uniform(duty_low, duty_high), 1e-3, 1.0)
    span = max(1, int(round(T * duty_cycle)))
    start = rng.integers(0, T)
    idx = (np.arange(span) + start) % T
    jam = sigma * (rng.standard_normal(sample[idx, :].shape) + 1j * rng.standard_normal(sample[idx, :].shape))
    sample[idx, :] = sample[idx, :] + jam
    mask[idx, :] = np.maximum(mask[idx, :], 1.0)
    return sample, mask


def _apply_fh(sample, mask, jsr_db, rng, K, group, cfg):
    sigma = _compute_jam_sigma(sample, jsr_db)
    sample_view = sample.reshape(sample.shape[0], K, group)
    mask_view = mask.reshape(mask.shape[0], K, group)
    hop_low, hop_high = cfg['fh_hop_len_range']
    hop_len = int(rng.integers(max(1, hop_low), max(2, hop_high + 1)))
    bw_indices = _select_band(rng, K, cfg['fh_bandwidth_frac_range'])
    for start in range(0, sample.shape[0], hop_len):
        stop = min(start + hop_len, sample.shape[0])
        hopping_band = np.copy(bw_indices)
        rng.shuffle(hopping_band)
        size = rng.integers(1, len(hopping_band) + 1)
        active = hopping_band[:size]
        jam_shape = (stop - start, active.shape[0], group)
        jam = sigma * (rng.standard_normal(jam_shape) + 1j * rng.standard_normal(jam_shape))
        sample_view[start:stop, active, :] = sample_view[start:stop, active, :] + jam
        mask_view[start:stop, active, :] = np.maximum(mask_view[start:stop, active, :], 1.0)
    return sample_view.reshape(sample.shape), mask_view.reshape(mask.shape)


def apply_jammers(H, cfg=None):
    if cfg is None:
        return H, np.zeros(H.shape, dtype=np.float32)
    cfg = _merge_jammer_cfg(cfg)
    rng = _ensure_rng(cfg.get('seed'))
    H_out = H.copy()
    mask = np.zeros(H.shape, dtype=np.float32)
    B, T, mul = H.shape
    K = cfg.get('num_subcarriers', DEFAULT_JAMMER_CFG['num_subcarriers'])
    if mul % K != 0:
        raise ValueError(f"Invalid jammer configuration: feature dimension {mul} is not divisible by subcarriers {K}")
    group = mul // K
    jammer_types = cfg.get('types', DEFAULT_JAMMER_CFG['types'])
    jam_prob = float(cfg.get('jam_prob', DEFAULT_JAMMER_CFG['jam_prob']))
    jam_prob = max(0.0, min(1.0, jam_prob))
    min_types = max(0, int(cfg.get('min_types', DEFAULT_JAMMER_CFG['min_types'])))
    max_types = max(min_types, int(cfg.get('max_types', DEFAULT_JAMMER_CFG['max_types'])))
    if not jammer_types:
        return H_out, mask
    for b in range(B):
        if rng.random() > jam_prob:
            continue
        if max_types <= 0:
            continue
        if min_types >= len(jammer_types):
            num_types = len(jammer_types)
        elif min_types == max_types:
            num_types = min(min_types, len(jammer_types))
        else:
            num_types = int(rng.integers(min_types, max_types + 1))
            num_types = max(min_types, num_types)
            num_types = min(num_types, len(jammer_types))
        if num_types <= 0:
            continue
        active_types = rng.choice(jammer_types, size=num_types, replace=False)
        for jammer in active_types:
            jsr_db = rng.uniform(*cfg['jsr_db_range'])
            if jammer == 'wb':
                H_out[b], mask[b] = _apply_wb(H_out[b], mask[b], jsr_db, rng)
            elif jammer == 'pb':
                H_out[b], mask[b] = _apply_pb(H_out[b], mask[b], jsr_db, rng, K, group, cfg)
            elif jammer == 'tone':
                H_out[b], mask[b] = _apply_tone(H_out[b], mask[b], jsr_db, rng, K, group, cfg)
            elif jammer == 'pulse':
                H_out[b], mask[b] = _apply_pulse(H_out[b], mask[b], jsr_db, rng, cfg)
            elif jammer == 'fh':
                H_out[b], mask[b] = _apply_fh(H_out[b], mask[b], jsr_db, rng, K, group, cfg)
            else:
                raise ValueError(f"Unsupported jammer type: {jammer}")
    return H_out, mask


def LoadBatch_mask(mask, num=32):
    B, T, mul = mask.shape
    mask = rearrange(mask, 'b t (k a) -> (b a) t k', a=num)
    mask = np.repeat(mask, 2, axis=-1)
    return torch.tensor(mask, dtype=torch.float32)


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_r, file_path_t, is_train=1, ir=1, SNR=15, is_U2D=0, is_few=0,
                 train_per=0.9, valid_per=0.1, use_jammer=False, jammer_cfg=None, return_mask=False):
        super(Dataset_Pro, self).__init__()
        self.SNR = SNR
        self.ir = ir
        self.use_jammer = use_jammer
        self.return_mask = return_mask and use_jammer
        self.jammer_cfg = _merge_jammer_cfg(jammer_cfg) if use_jammer else None
        H_his = hdf5storage.loadmat(file_path_r)['H_U_his_train']  # v,b,l,k,a,b,c
        if is_U2D:
            H_pre = hdf5storage.loadmat(file_path_t)["H_D_pre_train"]  # v,b,l,k,a,b,c
        else:
            H_pre = hdf5storage.loadmat(file_path_t)["H_U_pre_train"]  # v,b,l,k,a,b,c
        # print(H_his.shape, H_pre.shape)

        batch = H_pre.shape[1]
        if is_train:
            H_his = H_his[:, :int(train_per * batch), ...]
            H_pre = H_pre[:, :int(train_per * batch), ...]
        else:
            H_his = H_his[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
            H_pre = H_pre[:, int(train_per * batch):int((train_per + valid_per) * batch), ...]
        H_his = rearrange(H_his, 'v n L k a b c -> (v n) L (k a b c)')
        H_pre = rearrange(H_pre, 'v n L k a b c -> (v n) L (k a b c)')

        B, prev_len, mul = H_his.shape
        _, pred_len, mul = H_pre.shape
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.seq_len = pred_len + prev_len

        dt_all = np.concatenate((H_his, H_pre), axis=1)
        np.random.shuffle(dt_all)
        H_his = dt_all[:, :prev_len, ...]
        H_pre = dt_all[:, -pred_len:, ...]
        for i in range(B):
            H_his[i, ...] = noise(H_his[i, ...], random.rand() * 15 + 5.0)
            H_pre[i, ...] = noise(H_pre[i, ...], random.rand() * 15 + 5.0)
        jam_mask = np.zeros_like(H_his.real, dtype=np.float32)
        if self.use_jammer:
            H_his, jam_mask = apply_jammers(H_his, self.jammer_cfg)
        std = np.sqrt(np.std(np.abs(H_his) ** 2))
        H_his = H_his / std
        H_pre = H_pre / std
        H_pre = LoadBatch_ofdm(H_pre)
        H_his = LoadBatch_ofdm(H_his)
        if is_few == 1:
            H_pre = H_pre[::10, ...]
            H_his = H_his[::10, ...]
        if self.use_jammer:
            jam_mask = LoadBatch_mask(jam_mask)
            if is_few == 1:
                jam_mask = jam_mask[::10, ...]
            self.jam_mask = jam_mask
        else:
            self.jam_mask = None
        self.pred = H_pre  # b,16,(48*2)
        self.prev = H_his  # b,4,(48*2)

    def __getitem__(self, index):
        if self.return_mask:
            return (self.pred[index, :].float(),
                    self.prev[index, :].float(),
                    self.jam_mask[index, :].float())
        return self.pred[index, :].float(), \
               self.prev[index, :].float()

    def __len__(self):
        return self.pred.shape[0]


def LoadBatch_ofdm_2(H):
    # H: B,T,K,mul     [tensor complex]
    # out:B,T,K,mul*2  [tensor real]
    B, T, K, mul = H.shape
    H_real = np.zeros([B, T, K, mul, 2])
    H_real[:, :, :, :, 0] = H.real
    H_real[:, :, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, K, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm_1(H):
    # H: B,T,mul     [tensor complex]
    # out:B,T,mul*2  [tensor real]
    B, T, mul = H.shape
    H_real = np.zeros([B, T, mul, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B, T, mul * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def LoadBatch_ofdm(H, num=32):
    # H: B,T,mul             [tensor complex]
    # out:B*num,T,mul*2/num  [tensor real]
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) ->(b a) t k', a=num)
    H_real = np.zeros([B * num, T, mul // num, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, mul // num * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def Transform_TDD_FDD(H, Nt=4, Nr=4):
    # H: B,T,mul    [tensor real]
    # out:B',Nt,Nr  [tensor complex]
    H = H.reshape(-1, Nt, Nr, 2)
    H_real = H[..., 0]
    H_imag = H[..., 1]
    out = torch.complex(H_real, H_imag)
    return out
