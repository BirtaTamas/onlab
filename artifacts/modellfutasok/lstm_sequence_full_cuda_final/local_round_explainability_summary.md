# Local Round Explainability

- csv_path: `processed_full\iem_chengdu\iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR\heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `21`

## Largest probability jumps

- tick `167130`, seconds `49.50`, LSTM `0.0922`, delta `-0.6591`
- tick `166138`, seconds `34.00`, LSTM `0.7209`, delta `+0.1957`
- tick `165562`, seconds `25.00`, LSTM `0.5683`, delta `+0.1855`
- tick `167066`, seconds `48.50`, LSTM `0.7869`, delta `-0.1252`
- tick `167034`, seconds `48.00`, LSTM `0.9121`, delta `+0.1072`
- tick `165338`, seconds `21.50`, LSTM `0.4455`, delta `-0.0803`
- tick `166170`, seconds `34.50`, LSTM `0.7934`, delta `+0.0726`
- tick `167162`, seconds `50.00`, LSTM `0.0270`, delta `-0.0653`
- tick `165402`, seconds `22.50`, LSTM `0.3170`, delta `-0.0648`
- tick `165370`, seconds `22.00`, LSTM `0.3818`, delta `-0.0637`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.005140`, |coef| `0.005140`
- `lag_00__kill_diff_last_3s`: coefficient `0.005054`, |coef| `0.005054`
- `lag_00__damage_diff_last_5s`: coefficient `0.004780`, |coef| `0.004780`
- `lag_00__T_damage_last_5s`: coefficient `-0.004237`, |coef| `0.004237`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003942`, |coef| `0.003942`
- `lag_11__CT5__is_scoped`: coefficient `-0.003786`, |coef| `0.003786`
- `lag_07__T_bomb_zone_count`: coefficient `-0.003530`, |coef| `0.003530`
- `lag_03__T_place_ALLEY`: coefficient `0.003476`, |coef| `0.003476`
- `lag_10__CT_place_SIDEHALL`: coefficient `-0.003236`, |coef| `0.003236`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003156`, |coef| `0.003156`
- `lag_00__CT_place_SIDEHALL`: coefficient `0.003079`, |coef| `0.003079`
- `lag_06__CT5__duck_amount`: coefficient `0.003016`, |coef| `0.003016`
- `lag_12__T4__duck_amount`: coefficient `-0.002991`, |coef| `0.002991`
- `lag_09__T4__duck_amount`: coefficient `0.002737`, |coef| `0.002737`
- `lag_03__CT5__is_scoped`: coefficient `0.002695`, |coef| `0.002695`
