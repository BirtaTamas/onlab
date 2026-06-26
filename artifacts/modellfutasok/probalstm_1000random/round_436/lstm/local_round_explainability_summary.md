# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `12`

## Largest probability jumps

- tick `97346`, seconds `65.00`, LSTM `0.8999`, delta `+0.2579`
- tick `97058`, seconds `60.50`, LSTM `0.5813`, delta `-0.1411`
- tick `96738`, seconds `55.50`, LSTM `0.6761`, delta `+0.1010`
- tick `97314`, seconds `64.50`, LSTM `0.6420`, delta `+0.0438`
- tick `96962`, seconds `59.00`, LSTM `0.7221`, delta `+0.0309`
- tick `97250`, seconds `63.50`, LSTM `0.5980`, delta `+0.0302`
- tick `96802`, seconds `56.50`, LSTM `0.7109`, delta `+0.0301`
- tick `97378`, seconds `65.50`, LSTM `0.9282`, delta `+0.0282`
- tick `97922`, seconds `74.00`, LSTM `0.9691`, delta `+0.0256`
- tick `93666`, seconds `7.50`, LSTM `0.6392`, delta `-0.0202`

## Top 15 local ridge features

- `lag_03__T_place_DECK`: coefficient `-0.001739`, |coef| `0.001739`
- `lag_12__T_place_BALCONY`: coefficient `0.001658`, |coef| `0.001658`
- `lag_08__T_place_BALCONY`: coefficient `-0.001649`, |coef| `0.001649`
- `lag_08__T_place_PIT`: coefficient `0.001370`, |coef| `0.001370`
- `lag_00__kill_diff_last_3s`: coefficient `0.001277`, |coef| `0.001277`
- `lag_00__CT_kills_last_3s`: coefficient `0.001224`, |coef| `0.001224`
- `lag_09__T_shots_fired_sum`: coefficient `0.001152`, |coef| `0.001152`
- `lag_10__CT4__flash_duration`: coefficient `-0.001100`, |coef| `0.001100`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001083`, |coef| `0.001083`
- `lag_15__T_place_BALCONY`: coefficient `0.001066`, |coef| `0.001066`
- `lag_00__T_place_PIT`: coefficient `-0.001053`, |coef| `0.001053`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001019`, |coef| `0.001019`
- `lag_09__T_place_PIT`: coefficient `0.000929`, |coef| `0.000929`
- `lag_00__damage_diff_last_5s`: coefficient `0.000928`, |coef| `0.000928`
- `lag_13__T_place_PIT`: coefficient `0.000903`, |coef| `0.000903`

## Top 10 utility ridge features

- `lag_10__CT4__flash_duration`: coefficient `-0.001100` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.000719` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `0.000667` (raises CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `-0.000660` (lowers CT win probability)
- `lag_12__CT4__flash_duration`: coefficient `-0.000638` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `-0.000546` (lowers CT win probability)
- `lag_12__T3__smoke`: coefficient `0.000541` (raises CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000525` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `-0.000506` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000471` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_DECK`: coefficient `-0.001739` (lowers CT win probability)
- `lag_12__T_place_BALCONY`: coefficient `0.001658` (raises CT win probability)
- `lag_08__T_place_BALCONY`: coefficient `-0.001649` (lowers CT win probability)
- `lag_08__T_place_PIT`: coefficient `0.001370` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001277` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001224` (raises CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `0.001152` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001083` (lowers CT win probability)
- `lag_15__T_place_BALCONY`: coefficient `0.001066` (raises CT win probability)
- `lag_00__T_place_PIT`: coefficient `-0.001053` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `97346`, seconds `65.00`, LSTM delta `+0.2579`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `+0.022803`
- `lag_08__T_place_BALCONY`: contribution `+0.022681`
- `lag_08__T_place_PIT`: contribution `+0.008645`
- `lag_09__T_shots_fired_sum`: contribution `+0.008635`
- `lag_03__T_place_BALCONY`: contribution `+0.007225`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `+0.006444`
- `lag_10__CT_flash_duration_sum`: contribution `+0.002778`

### tick `97058`, seconds `60.50`, LSTM delta `-0.1411`

Top all feature movements:
- `lag_08__T_place_BALCONY`: contribution `-0.022682`
- `lag_13__T_place_DECK`: contribution `-0.019519`
- `lag_00__T_shots_fired_sum`: contribution `-0.008116`
- `lag_03__T_place_BALCONY`: contribution `+0.007225`
- `lag_12__CT_flashed_players`: contribution `-0.003941`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `-0.003739`
- `lag_12__CT_flash_duration_sum`: contribution `-0.003176`
- `lag_12__CT3__flash_duration`: contribution `-0.002677`
- `lag_01__CT4__flash_duration`: contribution `-0.002589`

### tick `96738`, seconds `55.50`, LSTM delta `+0.1010`

Top all feature movements:
- `lag_03__T_place_DECK`: contribution `+0.042175`
- `lag_00__CT_kills_last_3s`: contribution `+0.003533`
- `lag_00__kill_diff_last_3s`: contribution `+0.003073`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002831`
- `lag_02__CT4__flash_duration`: contribution `+0.002756`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `+0.002756`
- `lag_02__CT3__flash_duration`: contribution `+0.002242`
- `lag_02__CT_flash_duration_sum`: contribution `+0.002042`

### tick `97314`, seconds `64.50`, LSTM delta `+0.0438`

Top all feature movements:
- `lag_11__T_place_BALCONY`: contribution `+0.009172`
- `lag_15__CT_place_QUAD`: contribution `+0.005426`
- `lag_07__T_shots_fired_sum`: contribution `+0.005367`
- `lag_08__T_shots_fired_sum`: contribution `-0.004928`
- `lag_00__T_shots_fired_sum`: contribution `-0.004058`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `+0.001538`

### tick `96962`, seconds `59.00`, LSTM delta `+0.0309`

Top all feature movements:
- `lag_10__T_place_DECK`: contribution `+0.010168`
- `lag_00__T_place_BALCONY`: contribution `+0.006378`
- `lag_05__T_place_BALCONY`: contribution `-0.004310`
- `lag_01__CT_place_ARCH`: contribution `+0.002226`
- `lag_08__CT1__shots_fired`: contribution `+0.001931`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.001538`
- `lag_09__CT_flash_duration_sum`: contribution `-0.000762`
