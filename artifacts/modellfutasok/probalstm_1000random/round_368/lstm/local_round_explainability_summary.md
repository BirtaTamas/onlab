# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `25`

## Largest probability jumps

- tick `209811`, seconds `86.50`, LSTM `0.5212`, delta `+0.3419`
- tick `205875`, seconds `25.00`, LSTM `0.1871`, delta `-0.3300`
- tick `210227`, seconds `93.00`, LSTM `0.1586`, delta `-0.2978`
- tick `211059`, seconds `106.00`, LSTM `0.3715`, delta `+0.1578`
- tick `210899`, seconds `103.50`, LSTM `0.1804`, delta `+0.1365`
- tick `205779`, seconds `23.50`, LSTM `0.4669`, delta `+0.1113`
- tick `205619`, seconds `21.00`, LSTM `0.3721`, delta `-0.1005`
- tick `205587`, seconds `20.50`, LSTM `0.4726`, delta `-0.0892`
- tick `205907`, seconds `25.50`, LSTM `0.1258`, delta `-0.0613`
- tick `209395`, seconds `80.00`, LSTM `0.2328`, delta `+0.0598`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004669`, |coef| `0.004669`
- `lag_04__CT1__flash_duration`: coefficient `0.004243`, |coef| `0.004243`
- `lag_00__CT_kills_last_3s`: coefficient `0.003421`, |coef| `0.003421`
- `lag_05__CT1__flash_duration`: coefficient `0.003161`, |coef| `0.003161`
- `lag_12__CT_place_JUNGLE`: coefficient `-0.003091`, |coef| `0.003091`
- `lag_00__T_place_TRAMP`: coefficient `-0.002829`, |coef| `0.002829`
- `lag_13__T_place_TRAMP`: coefficient `0.002580`, |coef| `0.002580`
- `lag_13__T_place_SCAFFOLDING`: coefficient `-0.002540`, |coef| `0.002540`
- `lag_00__damage_diff_last_5s`: coefficient `0.002454`, |coef| `0.002454`
- `lag_00__T_kills_last_3s`: coefficient `-0.002392`, |coef| `0.002392`
- `lag_13__CT_A_site_active_infernos`: coefficient `0.002035`, |coef| `0.002035`
- `lag_07__CT_kills_last_3s`: coefficient `0.002026`, |coef| `0.002026`
- `lag_09__CT1__duck_amount`: coefficient `0.002017`, |coef| `0.002017`
- `lag_04__CT_flash_duration_sum`: coefficient `0.001984`, |coef| `0.001984`
- `lag_13__CT_kills_last_3s`: coefficient `-0.001964`, |coef| `0.001964`

## Top 10 utility ridge features

- `lag_04__CT1__flash_duration`: coefficient `0.004243` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.003161` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `0.002035` (raises CT win probability)
- `lag_04__CT_flash_duration_sum`: coefficient `0.001984` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `-0.001944` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `-0.001858` (lowers CT win probability)
- `lag_13__CT_B_site_active_infernos`: coefficient `0.001852` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `0.001799` (raises CT win probability)
- `lag_03__CT_utility_damage_last_5s`: coefficient `-0.001762` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `0.001714` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004669` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003421` (raises CT win probability)
- `lag_12__CT_place_JUNGLE`: coefficient `-0.003091` (lowers CT win probability)
- `lag_00__T_place_TRAMP`: coefficient `-0.002829` (lowers CT win probability)
- `lag_13__T_place_TRAMP`: coefficient `0.002580` (raises CT win probability)
- `lag_13__T_place_SCAFFOLDING`: coefficient `-0.002540` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002454` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002392` (lowers CT win probability)
- `lag_07__CT_kills_last_3s`: coefficient `0.002026` (raises CT win probability)
- `lag_09__CT1__duck_amount`: coefficient `0.002017` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `209811`, seconds `86.50`, LSTM delta `+0.3419`

Top all feature movements:
- `lag_04__CT1__flash_duration`: contribution `+0.027345`
- `lag_00__kill_diff_last_3s`: contribution `+0.022477`
- `lag_12__CT_place_JUNGLE`: contribution `+0.019830`
- `lag_00__CT_kills_last_3s`: contribution `+0.019752`
- `lag_00__T_place_TRAMP`: contribution `+0.016562`

Top utility-only movements:
- `lag_04__CT1__flash_duration`: contribution `+0.027345`
- `lag_13__CT_A_site_active_infernos`: contribution `+0.007181`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.006362`
- `lag_04__CT_flash_duration_sum`: contribution `+0.005744`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.005571`

### tick `205875`, seconds `25.00`, LSTM delta `-0.3300`

Top all feature movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.012800`
- `lag_08__CT_flashed_players`: contribution `-0.012203`
- `lag_09__CT_shots_fired_sum`: contribution `-0.012154`
- `lag_00__kill_diff_last_3s`: contribution `-0.011238`
- `lag_08__CT3__flash_duration`: contribution `-0.009981`

Top utility-only movements:
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.012800`
- `lag_08__CT3__flash_duration`: contribution `-0.009981`
- `lag_08__CT2__flash_duration`: contribution `-0.009738`
- `lag_08__CT_flash_duration_sum`: contribution `-0.007587`
- `lag_08__CT4__flash_duration`: contribution `-0.005769`

### tick `210227`, seconds `93.00`, LSTM delta `-0.2978`

Top all feature movements:
- `lag_05__CT1__flash_duration`: contribution `-0.020373`
- `lag_13__T_place_TRAMP`: contribution `-0.015100`
- `lag_07__CT_kills_last_3s`: contribution `-0.011699`
- `lag_13__CT_kills_last_3s`: contribution `-0.011341`
- `lag_00__kill_diff_last_3s`: contribution `-0.011238`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.020373`
- `lag_15__CT_A_site_active_infernos`: contribution `-0.006048`
- `lag_05__CT_flash_duration_sum`: contribution `-0.005209`
- `lag_02__T_utility_damage_last_5s`: contribution `-0.003806`

### tick `211059`, seconds `106.00`, LSTM delta `+0.1578`

Top all feature movements:
- `lag_13__T_place_SCAFFOLDING`: contribution `+0.086505`
- `lag_06__CT_shots_fired_sum`: contribution `+0.006346`
- `lag_07__T4__duck_amount`: contribution `+0.004883`
- `lag_11__T_place_CONNECTOR`: contribution `+0.003644`
- `lag_07__CT_duck_amount_mean`: contribution `+0.003512`

Top utility-only movements:
- `lag_12__T2__utility_total`: contribution `+0.001940`
- `lag_12__T2__flash`: contribution `+0.001693`

### tick `210899`, seconds `103.50`, LSTM delta `+0.1365`

Top all feature movements:
- `lag_08__T_place_SCAFFOLDING`: contribution `+0.060596`
- `lag_00__kill_diff_last_3s`: contribution `+0.011238`
- `lag_12__T_place_SCAFFOLDING`: contribution `-0.010579`
- `lag_00__CT_kills_last_3s`: contribution `+0.009876`
- `lag_07__CT_kills_last_3s`: contribution `+0.005850`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `+0.003110`
