# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `11`

## Largest probability jumps

- tick `88013`, seconds `92.00`, LSTM `0.7387`, delta `+0.4437`
- tick `85677`, seconds `55.50`, LSTM `0.1146`, delta `-0.3782`
- tick `88109`, seconds `93.50`, LSTM `0.5313`, delta `-0.3083`
- tick `87437`, seconds `83.00`, LSTM `0.3160`, delta `+0.2305`
- tick `88205`, seconds `95.00`, LSTM `0.9028`, delta `+0.1724`
- tick `84525`, seconds `37.50`, LSTM `0.4892`, delta `-0.1324`
- tick `87373`, seconds `82.00`, LSTM `0.1657`, delta `+0.1077`
- tick `88173`, seconds `94.50`, LSTM `0.7305`, delta `+0.1042`
- tick `88141`, seconds `94.00`, LSTM `0.6262`, delta `+0.0950`
- tick `88077`, seconds `93.00`, LSTM `0.8396`, delta `+0.0870`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007141`, |coef| `0.007141`
- `lag_00__CT_shots_fired_sum`: coefficient `0.006684`, |coef| `0.006684`
- `lag_11__T_place_LONGDOG`: coefficient `-0.006485`, |coef| `0.006485`
- `lag_00__CT_kills_last_3s`: coefficient `0.005255`, |coef| `0.005255`
- `lag_00__T_shots_fired_sum`: coefficient `-0.004897`, |coef| `0.004897`
- `lag_13__CT1__smoke`: coefficient `0.004622`, |coef| `0.004622`
- `lag_00__CT5__is_scoped`: coefficient `-0.004570`, |coef| `0.004570`
- `lag_00__damage_diff_last_5s`: coefficient `0.004443`, |coef| `0.004443`
- `lag_00__T_spread_xy`: coefficient `-0.004019`, |coef| `0.004019`
- `lag_00__T1__alive`: coefficient `-0.003955`, |coef| `0.003955`
- `lag_01__T1__shots_fired`: coefficient `0.003942`, |coef| `0.003942`
- `lag_00__T1__hp`: coefficient `-0.003892`, |coef| `0.003892`
- `lag_01__CT2__shots_fired`: coefficient `-0.003886`, |coef| `0.003886`
- `lag_00__CT_damage_last_5s`: coefficient `0.003851`, |coef| `0.003851`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003741`, |coef| `0.003741`

## Top 10 utility ridge features

- `lag_13__CT1__smoke`: coefficient `0.004622` (raises CT win probability)
- `lag_00__T_flash_alpha_mean`: coefficient `-0.003741` (lowers CT win probability)
- `lag_10__CT1__smoke`: coefficient `-0.002805` (lowers CT win probability)
- `lag_07__CT_B_site_active_smokes`: coefficient `0.002727` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.002007` (lowers CT win probability)
- `lag_13__CT1__utility_total`: coefficient `0.001931` (raises CT win probability)
- `lag_07__CT_active_smokes`: coefficient `0.001906` (raises CT win probability)
- `lag_12__T5__smoke`: coefficient `0.001507` (raises CT win probability)
- `lag_13__CT_smoke_inv`: coefficient `0.001471` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001321` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.007141` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.006684` (raises CT win probability)
- `lag_11__T_place_LONGDOG`: coefficient `-0.006485` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.005255` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.004897` (lowers CT win probability)
- `lag_00__CT5__is_scoped`: coefficient `-0.004570` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004443` (raises CT win probability)
- `lag_00__T_spread_xy`: coefficient `-0.004019` (lowers CT win probability)
- `lag_00__T1__alive`: coefficient `-0.003955` (lowers CT win probability)
- `lag_01__T1__shots_fired`: coefficient `0.003942` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `88013`, seconds `92.00`, LSTM delta `+0.4437`

Top all feature movements:
- `lag_11__T_place_LONGDOG`: contribution `+0.030178`
- `lag_00__kill_diff_last_3s`: contribution `+0.017187`
- `lag_00__CT5__is_scoped`: contribution `+0.016343`
- `lag_00__CT_kills_last_3s`: contribution `+0.015173`
- `lag_00__T_shots_fired_sum`: contribution `+0.014685`

Top utility-only movements:
- `lag_13__CT1__smoke`: contribution `+0.010019`

### tick `85677`, seconds `55.50`, LSTM delta `-0.3782`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.041790`
- `lag_12__T_place_DUMPSTER`: contribution `-0.031789`
- `lag_04__T_place_DUMPSTER`: contribution `-0.019653`
- `lag_00__T_shots_fired_sum`: contribution `-0.018356`
- `lag_00__kill_diff_last_3s`: contribution `-0.017187`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88109`, seconds `93.50`, LSTM delta `-0.3083`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.027860`
- `lag_00__kill_diff_last_3s`: contribution `-0.017187`
- `lag_00__T_duck_amount_mean`: contribution `-0.013085`
- `lag_00__CT4__duck_amount`: contribution `-0.012386`
- `lag_00__T_kills_last_3s`: contribution `-0.011507`

Top utility-only movements:
- `lag_13__CT1__smoke`: contribution `-0.010019`

### tick `87437`, seconds `83.00`, LSTM delta `+0.2305`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.024524`
- `lag_00__CT_shots_fired_sum`: contribution `+0.023217`
- `lag_00__kill_diff_last_3s`: contribution `+0.017187`
- `lag_00__CT_kills_last_3s`: contribution `+0.015173`
- `lag_00__T_shots_fired_sum`: contribution `+0.011013`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88205`, seconds `95.00`, LSTM delta `+0.1724`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.022699`
- `lag_00__CT_shots_fired_sum`: contribution `-0.013930`
- `lag_09__CT1__is_walking`: contribution `+0.008365`
- `lag_02__T_duck_amount_mean`: contribution `+0.007129`
- `lag_06__CT5__is_scoped`: contribution `+0.006778`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.022699`
