# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-liquid-vs-mouz-bo3-heKnTsZGq8rrQ4y9Yn2KrI/liquid-vs-mouz-m2-train.csv`
- round_num: `15`

## Largest probability jumps

- tick `98643`, seconds `10.00`, LSTM `0.5352`, delta `-0.1796`
- tick `98835`, seconds `13.00`, LSTM `0.6778`, delta `+0.1778`
- tick `99955`, seconds `30.50`, LSTM `0.8497`, delta `+0.1107`
- tick `100083`, seconds `32.50`, LSTM `0.9506`, delta `+0.0606`
- tick `99635`, seconds `25.50`, LSTM `0.7552`, delta `+0.0573`
- tick `99027`, seconds `16.00`, LSTM `0.5984`, delta `-0.0539`
- tick `98579`, seconds `9.00`, LSTM `0.6928`, delta `+0.0474`
- tick `99539`, seconds `24.00`, LSTM `0.7111`, delta `+0.0470`
- tick `98803`, seconds `12.50`, LSTM `0.5000`, delta `-0.0444`
- tick `98867`, seconds `13.50`, LSTM `0.6340`, delta `-0.0438`

## Top 15 local ridge features

- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.001504`, |coef| `0.001504`
- `lag_14__CT_place_ENTRANCE`: coefficient `0.001463`, |coef| `0.001463`
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001443`, |coef| `0.001443`
- `lag_00__kill_diff_last_3s`: coefficient `0.001389`, |coef| `0.001389`
- `lag_07__CT_place_ELECTRICALBOX`: coefficient `0.001373`, |coef| `0.001373`
- `lag_06__CT_place_ELECTRICALBOX`: coefficient `-0.001293`, |coef| `0.001293`
- `lag_09__T_place_TMAIN`: coefficient `-0.001210`, |coef| `0.001210`
- `lag_00__CT_place_IVY`: coefficient `0.001045`, |coef| `0.001045`
- `lag_01__T_place_DUMPSTER`: coefficient `-0.000983`, |coef| `0.000983`
- `lag_10__T_place_TMAIN`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_00__CT_kills_last_3s`: coefficient `0.000971`, |coef| `0.000971`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000852`, |coef| `0.000852`
- `lag_00__CT2__flash_duration`: coefficient `-0.000821`, |coef| `0.000821`
- `lag_12__CT_place_ENTRANCE`: coefficient `0.000808`, |coef| `0.000808`
- `lag_10__CT_place_IVY`: coefficient `0.000796`, |coef| `0.000796`

## Top 10 utility ridge features

- `lag_00__CT2__flash_duration`: coefficient `-0.000821` (lowers CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `-0.000696` (lowers CT win probability)
- `lag_08__CT_utility_damage_last_5s`: coefficient `0.000641` (raises CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000634` (lowers CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.000589` (raises CT win probability)
- `lag_02__utility_damage_diff_last_5s`: coefficient `-0.000562` (lowers CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `-0.000553` (lowers CT win probability)
- `lag_08__utility_damage_diff_last_5s`: coefficient `0.000533` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.000489` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000463` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.001504` (lowers CT win probability)
- `lag_14__CT_place_ENTRANCE`: coefficient `0.001463` (raises CT win probability)
- `lag_00__CT_place_ELECTRICALBOX`: coefficient `0.001443` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001389` (raises CT win probability)
- `lag_07__CT_place_ELECTRICALBOX`: coefficient `0.001373` (raises CT win probability)
- `lag_06__CT_place_ELECTRICALBOX`: coefficient `-0.001293` (lowers CT win probability)
- `lag_09__T_place_TMAIN`: coefficient `-0.001210` (lowers CT win probability)
- `lag_00__CT_place_IVY`: coefficient `0.001045` (raises CT win probability)
- `lag_01__T_place_DUMPSTER`: coefficient `-0.000983` (lowers CT win probability)
- `lag_10__T_place_TMAIN`: coefficient `-0.000972` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `98643`, seconds `10.00`, LSTM delta `-0.1796`

Top all feature movements:
- `lag_01__CT_place_ELECTRICALBOX`: contribution `-0.017478`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `-0.016772`
- `lag_14__CT_place_ENTRANCE`: contribution `-0.012983`
- `lag_09__T_place_TMAIN`: contribution `-0.009389`
- `lag_12__CT_place_ENTRANCE`: contribution `-0.007165`

Top utility-only movements:
- `lag_02__CT_utility_damage_last_5s`: contribution `-0.005898`
- `lag_00__CT2__flash_duration`: contribution `-0.004838`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.003908`
- `lag_00__CT5__flash_duration`: contribution `-0.003292`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002329`

### tick `98835`, seconds `13.00`, LSTM delta `+0.1778`

Top all feature movements:
- `lag_07__CT_place_ELECTRICALBOX`: contribution `+0.015963`
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.015034`
- `lag_00__kill_diff_last_3s`: contribution `+0.006687`
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.005436`
- `lag_12__T_place_TSTAIRS`: contribution `+0.003756`

Top utility-only movements:
- `lag_08__CT_utility_damage_last_5s`: contribution `+0.005436`
- `lag_08__utility_damage_diff_last_5s`: contribution `+0.003706`
- `lag_06__CT2__flash_duration`: contribution `+0.003468`
- `lag_06__CT5__flash_duration`: contribution `+0.002539`

### tick `99955`, seconds `30.50`, LSTM delta `+0.1107`

Top all feature movements:
- `lag_10__CT_place_IVY`: contribution `+0.009079`
- `lag_13__CT_place_IVY`: contribution `+0.007642`
- `lag_11__T_place_DUMPSTER`: contribution `+0.005782`
- `lag_10__T_place_TMAIN`: contribution `+0.003771`
- `lag_09__T_place_DUMPSTER`: contribution `+0.003612`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `100083`, seconds `32.50`, LSTM delta `+0.0606`

Top all feature movements:
- `lag_14__CT_place_IVY`: contribution `+0.007186`
- `lag_00__kill_diff_last_3s`: contribution `+0.003344`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002960`
- `lag_15__T_place_DUMPSTER`: contribution `-0.002942`
- `lag_00__CT_kills_last_3s`: contribution `+0.002803`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `99635`, seconds `25.50`, LSTM delta `+0.0573`

Top all feature movements:
- `lag_00__CT_place_IVY`: contribution `+0.011931`
- `lag_01__T_place_DUMPSTER`: contribution `+0.008934`
- `lag_07__T_place_DUMPSTER`: contribution `+0.003862`
- `lag_02__CT4__duck_amount`: contribution `+0.002022`
- `lag_00__CT5__duck_amount`: contribution `+0.001551`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.000900`
