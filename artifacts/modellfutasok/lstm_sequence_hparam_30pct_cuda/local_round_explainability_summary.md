# Local Round Explainability

- csv_path: `processed_full\blast_bounty_season_1_finals\blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi\spirit-vs-heroic-m2-nuke.csv`
- round_num: `15`

## Largest probability jumps

- tick `128898`, seconds `105.00`, LSTM `0.9268`, delta `+0.8289`
- tick `126370`, seconds `65.50`, LSTM `0.2481`, delta `-0.3059`
- tick `123426`, seconds `19.50`, LSTM `0.4626`, delta `-0.1716`
- tick `124290`, seconds `33.00`, LSTM `0.3322`, delta `-0.1443`
- tick `124802`, seconds `41.00`, LSTM `0.3860`, delta `-0.0973`
- tick `126114`, seconds `61.50`, LSTM `0.5433`, delta `+0.0883`
- tick `124450`, seconds `35.50`, LSTM `0.3500`, delta `+0.0793`
- tick `124578`, seconds `37.50`, LSTM `0.4833`, delta `+0.0751`
- tick `126018`, seconds `60.00`, LSTM `0.4550`, delta `+0.0744`
- tick `123522`, seconds `21.00`, LSTM `0.3938`, delta `-0.0688`

## Top 15 local ridge features

- `lag_08__CT_place_LOBBY`: coefficient `-0.002362`, |coef| `0.002362`
- `lag_12__CT_place_LOBBY`: coefficient `-0.002251`, |coef| `0.002251`
- `lag_09__CT_place_LOBBY`: coefficient `-0.002185`, |coef| `0.002185`
- `lag_11__CT_place_LOBBY`: coefficient `-0.002185`, |coef| `0.002185`
- `lag_10__CT_place_LOBBY`: coefficient `-0.002185`, |coef| `0.002185`
- `lag_03__CT_shots_fired_sum`: coefficient `0.001984`, |coef| `0.001984`
- `lag_00__kill_diff_last_3s`: coefficient `0.001925`, |coef| `0.001925`
- `lag_01__kill_diff_last_3s`: coefficient `0.001885`, |coef| `0.001885`
- `lag_01__CT_kills_last_3s`: coefficient `0.001845`, |coef| `0.001845`
- `lag_02__CT_kills_last_3s`: coefficient `0.001824`, |coef| `0.001824`
- `lag_00__CT_kills_last_3s`: coefficient `0.001824`, |coef| `0.001824`
- `lag_12__CT_place_HEAVEN`: coefficient `0.001808`, |coef| `0.001808`
- `lag_02__kill_diff_last_3s`: coefficient `0.001779`, |coef| `0.001779`
- `lag_02__damage_diff_last_5s`: coefficient `0.001752`, |coef| `0.001752`
- `lag_11__CT_place_HEAVEN`: coefficient `0.001746`, |coef| `0.001746`
