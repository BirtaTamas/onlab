# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `14`

## Largest probability jumps

- tick `128195`, seconds `76.00`, LSTM `0.3291`, delta `-0.2000`
- tick `128003`, seconds `73.00`, LSTM `0.4656`, delta `+0.1888`
- tick `128483`, seconds `80.50`, LSTM `0.0656`, delta `-0.1329`
- tick `128675`, seconds `83.50`, LSTM `0.0810`, delta `-0.1324`
- tick `128227`, seconds `76.50`, LSTM `0.2031`, delta `-0.1259`
- tick `127939`, seconds `72.00`, LSTM `0.3005`, delta `-0.0996`
- tick `128547`, seconds `81.50`, LSTM `0.1795`, delta `+0.0996`
- tick `127459`, seconds `64.50`, LSTM `0.4750`, delta `+0.0969`
- tick `127363`, seconds `63.00`, LSTM `0.3570`, delta `-0.0736`
- tick `126531`, seconds `50.00`, LSTM `0.3759`, delta `+0.0725`

## Top 15 local ridge features

- `lag_00__T_place_SIDE`: coefficient `-0.003363`, |coef| `0.003363`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.002234`, |coef| `0.002234`
- `lag_00__T_flashes_last_5s`: coefficient `-0.002035`, |coef| `0.002035`
- `lag_06__T_place_SIDE`: coefficient `0.002023`, |coef| `0.002023`
- `lag_09__T3__is_scoped`: coefficient `-0.001850`, |coef| `0.001850`
- `lag_09__T_place_SIDE`: coefficient `-0.001621`, |coef| `0.001621`
- `lag_02__T_place_SIDE`: coefficient `0.001466`, |coef| `0.001466`
- `lag_10__T3__is_scoped`: coefficient `-0.001434`, |coef| `0.001434`
- `lag_04__T_place_SIDE`: coefficient `0.001409`, |coef| `0.001409`
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `-0.001402`, |coef| `0.001402`
- `lag_11__T3__is_scoped`: coefficient `-0.001402`, |coef| `0.001402`
- `lag_00__CT_place_EXTENDEDA`: coefficient `0.001211`, |coef| `0.001211`
- `lag_04__CT_shots_fired_sum`: coefficient `0.001199`, |coef| `0.001199`
- `lag_08__T_place_SIDE`: coefficient `-0.001184`, |coef| `0.001184`
- `lag_05__T1__duck_amount`: coefficient `0.001183`, |coef| `0.001183`

## Top 10 utility ridge features

- `lag_00__T_flashes_last_5s`: coefficient `-0.002035` (lowers CT win probability)
- `lag_11__T_flash_duration_sum`: coefficient `0.001127` (raises CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `-0.000950` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.000924` (lowers CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.000906` (raises CT win probability)
- `lag_11__T1__flash_duration`: coefficient `0.000900` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000890` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000821` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000817` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.000800` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDE`: coefficient `-0.003363` (lowers CT win probability)
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.002234` (raises CT win probability)
- `lag_06__T_place_SIDE`: coefficient `0.002023` (raises CT win probability)
- `lag_09__T3__is_scoped`: coefficient `-0.001850` (lowers CT win probability)
- `lag_09__T_place_SIDE`: coefficient `-0.001621` (lowers CT win probability)
- `lag_02__T_place_SIDE`: coefficient `0.001466` (raises CT win probability)
- `lag_10__T3__is_scoped`: coefficient `-0.001434` (lowers CT win probability)
- `lag_04__T_place_SIDE`: coefficient `0.001409` (raises CT win probability)
- `lag_00__CT_place_UPPERTUNNEL`: coefficient `-0.001402` (lowers CT win probability)
- `lag_11__T3__is_scoped`: coefficient `-0.001402` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `128195`, seconds `76.00`, LSTM delta `-0.2000`

Top all feature movements:
- `lag_06__T_place_SIDE`: contribution `-0.039148`
- `lag_08__T_place_SIDE`: contribution `-0.022908`
- `lag_09__T3__is_scoped`: contribution `-0.011864`
- `lag_04__CT_shots_fired_sum`: contribution `-0.011664`
- `lag_10__T3__is_scoped`: contribution `+0.009201`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `-0.008610`
- `lag_00__T5__flash_duration`: contribution `-0.006504`
- `lag_00__T2__flash_duration`: contribution `-0.006494`
- `lag_00__T1__flash_duration`: contribution `-0.003729`
- `lag_08__T1__flash_duration`: contribution `-0.003110`

### tick `128003`, seconds `73.00`, LSTM delta `+0.1888`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `+0.065067`
- `lag_02__T_place_SIDE`: contribution `+0.028363`
- `lag_09__T3__is_scoped`: contribution `+0.011864`
- `lag_11__T_flashed_players`: contribution `+0.007672`
- `lag_11__T_flash_duration_sum`: contribution `+0.006630`

Top utility-only movements:
- `lag_11__T_flash_duration_sum`: contribution `+0.006630`
- `lag_11__T1__flash_duration`: contribution `+0.004668`
- `lag_02__T1__flash_duration`: contribution `+0.003237`
- `lag_11__T3__flash_duration`: contribution `+0.003230`
- `lag_11__T2__flash_duration`: contribution `+0.002796`

### tick `128483`, seconds `80.50`, LSTM delta `-0.1329`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `-0.013330`
- `lag_15__T_place_SIDE`: contribution `-0.007675`
- `lag_08__T_place_ARAMP`: contribution `-0.007616`
- `lag_00__T_shots_fired_sum`: contribution `-0.007557`
- `lag_06__CT_shots_fired_sum`: contribution `-0.005458`

Top utility-only movements:
- `lag_09__T5__flash_duration`: contribution `-0.004162`
- `lag_02__CT3__flash_duration`: contribution `-0.002876`
- `lag_09__T2__flash_duration`: contribution `-0.002837`
- `lag_09__T_flash_duration_sum`: contribution `-0.002284`

### tick `128675`, seconds `83.50`, LSTM delta `-0.1324`

Top all feature movements:
- `lag_14__T_place_ARAMP`: contribution `-0.008586`
- `lag_10__CT_shots_fired_sum`: contribution `-0.007606`
- `lag_05__CT_place_LOWERTUNNEL`: contribution `-0.006274`
- `lag_05__T1__duck_amount`: contribution `-0.004632`
- `lag_14__T_place_LONGA`: contribution `-0.004535`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `-0.004509`
- `lag_15__T_flash_duration_sum`: contribution `-0.004143`
- `lag_15__T2__flash_duration`: contribution `-0.003482`
- `lag_08__CT3__flash_duration`: contribution `-0.003246`
- `lag_15__T5__flash_duration`: contribution `-0.003003`

### tick `128227`, seconds `76.50`, LSTM delta `-0.1259`

Top all feature movements:
- `lag_09__T_place_SIDE`: contribution `-0.031351`
- `lag_07__T_place_SIDE`: contribution `-0.015433`
- `lag_10__T3__is_scoped`: contribution `-0.009201`
- `lag_11__T3__is_scoped`: contribution `+0.008995`
- `lag_01__T_flash_duration_sum`: contribution `-0.007456`

Top utility-only movements:
- `lag_01__T_flash_duration_sum`: contribution `-0.007456`
- `lag_01__T2__flash_duration`: contribution `-0.005241`
- `lag_01__T5__flash_duration`: contribution `-0.004686`
- `lag_01__T1__flash_duration`: contribution `-0.003430`
