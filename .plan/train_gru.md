# TODO: GRU Baseline Training Script

- [x] Review existing `train_rnn.py`/`train_lstm.py` patterns and `models/model.py` GRU implementation.
- [x] Draft CLI covering GRU-specific knobs (hidden size, layers, grad clip, DataParallel) aligned with other baselines.
- [x] Implement `train_gru.py` mirroring shared utilities, logging, and checkpoint handling.
- [ ] Provide example launch commands (single-GPU, multi-GPU, FDD toggle).
