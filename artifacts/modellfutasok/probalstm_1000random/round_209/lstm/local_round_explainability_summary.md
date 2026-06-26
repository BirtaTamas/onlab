# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `15`

## Largest probability jumps

- tick `133347`, seconds `109.50`, LSTM `0.6871`, delta `+0.2972`
- tick `133027`, seconds `104.50`, LSTM `0.4870`, delta `-0.2763`
- tick `132547`, seconds `97.00`, LSTM `0.6017`, delta `-0.2443`
- tick `132451`, seconds `95.50`, LSTM `0.8242`, delta `+0.2243`
- tick `132579`, seconds `97.50`, LSTM `0.3876`, delta `-0.2142`
- tick `132643`, seconds `98.50`, LSTM `0.6822`, delta `+0.1890`
- tick `133443`, seconds `111.00`, LSTM `0.8520`, delta `+0.1826`
- tick `132611`, seconds `98.00`, LSTM `0.4931`, delta `+0.1056`
- tick `132323`, seconds `93.50`, LSTM `0.6064`, delta `+0.1034`
- tick `127939`, seconds `25.00`, LSTM `0.5978`, delta `+0.0854`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005343`, |coef| `0.005343`
- `lag_00__damage_diff_last_5s`: coefficient `0.004437`, |coef| `0.004437`
- `lag_00__CT_defusing_count`: coefficient `0.004300`, |coef| `0.004300`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004264`, |coef| `0.004264`
- `lag_00__CT_kills_last_3s`: coefficient `0.003995`, |coef| `0.003995`
- `lag_00__T_damage_last_5s`: coefficient `-0.003320`, |coef| `0.003320`
- `lag_03__CT_place_TSIDEUPPER`: coefficient `-0.002985`, |coef| `0.002985`
- `lag_00__CT_velocity_mean`: coefficient `-0.002852`, |coef| `0.002852`
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002683`, |coef| `0.002683`
- `lag_00__T_macro_B`: coefficient `-0.002682`, |coef| `0.002682`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002682`, |coef| `0.002682`
- `lag_00__T_kills_last_3s`: coefficient `-0.002648`, |coef| `0.002648`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `0.002498`, |coef| `0.002498`
- `lag_13__damage_diff_last_5s`: coefficient `-0.002363`, |coef| `0.002363`
- `lag_11__T_place_SIDEENTRANCE`: coefficient `-0.002298`, |coef| `0.002298`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004264` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.002683` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `0.001781` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001444` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001410` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001376` (raises CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `0.001339` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.001264` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.001166` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001151` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005343` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004437` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.004300` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003995` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.003320` (lowers CT win probability)
- `lag_03__CT_place_TSIDEUPPER`: coefficient `-0.002985` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002852` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002682` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002682` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002648` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `133347`, seconds `109.50`, LSTM delta `+0.2972`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.025868`
- `lag_00__kill_diff_last_3s`: contribution `+0.012860`
- `lag_02__T_duck_amount_mean`: contribution `+0.012716`
- `lag_00__CT_kills_last_3s`: contribution `+0.011535`
- `lag_00__damage_diff_last_5s`: contribution `+0.011411`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.025868`

### tick `133027`, seconds `104.50`, LSTM delta `-0.2763`

Top all feature movements:
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.018776`
- `lag_00__kill_diff_last_3s`: contribution `-0.012860`
- `lag_00__damage_diff_last_5s`: contribution `-0.010010`
- `lag_00__T_kills_last_3s`: contribution `-0.008390`
- `lag_00__T_damage_last_5s`: contribution `-0.007961`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `132547`, seconds `97.00`, LSTM delta `-0.2443`

Top all feature movements:
- `lag_03__CT_place_TSIDEUPPER`: contribution `-0.022435`
- `lag_00__damage_diff_last_5s`: contribution `-0.010310`
- `lag_00__T_damage_last_5s`: contribution `-0.008200`
- `lag_06__T_shots_fired_sum`: contribution `-0.007785`
- `lag_00__CT_flashed_players`: contribution `-0.006994`

Top utility-only movements:
- `lag_05__T3__flash_duration`: contribution `-0.003708`
- `lag_00__CT3__flash_duration`: contribution `-0.003100`

### tick `132451`, seconds `95.50`, LSTM delta `+0.2243`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.013069`
- `lag_00__kill_diff_last_3s`: contribution `+0.012860`
- `lag_00__CT_kills_last_3s`: contribution `+0.011535`
- `lag_02__T3__flash_duration`: contribution `+0.007223`
- `lag_02__T_flashed_players`: contribution `+0.007149`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.007223`
- `lag_03__CT_B_site_active_infernos`: contribution `+0.004602`
- `lag_02__T_flash_duration_sum`: contribution `+0.003287`
- `lag_02__T5__flash_duration`: contribution `+0.003084`
- `lag_06__CT1__molly`: contribution `+0.002745`

### tick `132579`, seconds `97.50`, LSTM delta `-0.2142`

Top all feature movements:
- `lag_04__CT_place_TSIDEUPPER`: contribution `-0.014921`
- `lag_00__kill_diff_last_3s`: contribution `-0.012860`
- `lag_00__T_kills_last_3s`: contribution `-0.008390`
- `lag_01__CT_flashed_players`: contribution `-0.007098`
- `lag_00__T_damage_last_5s`: contribution `-0.006448`

Top utility-only movements:
- No utility movement among the top local contributors.
