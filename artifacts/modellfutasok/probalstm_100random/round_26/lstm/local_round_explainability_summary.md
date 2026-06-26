# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `6963`, seconds `13.00`, LSTM `0.8391`, delta `+0.2191`
- tick `10995`, seconds `76.00`, LSTM `0.7804`, delta `+0.2149`
- tick `10931`, seconds `75.00`, LSTM `0.5615`, delta `-0.1574`
- tick `11251`, seconds `80.00`, LSTM `0.9373`, delta `+0.1017`
- tick `6931`, seconds `12.50`, LSTM `0.6200`, delta `+0.0896`
- tick `11539`, seconds `84.50`, LSTM `0.8753`, delta `-0.0646`
- tick `11219`, seconds `79.50`, LSTM `0.8356`, delta `+0.0539`
- tick `11571`, seconds `85.00`, LSTM `0.8292`, delta `-0.0461`
- tick `7315`, seconds `18.50`, LSTM `0.7680`, delta `-0.0419`
- tick `11667`, seconds `86.50`, LSTM `0.8044`, delta `-0.0352`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002305`, |coef| `0.002305`
- `lag_00__kill_diff_last_3s`: coefficient `0.002175`, |coef| `0.002175`
- `lag_01__T_place_ARCH`: coefficient `0.001950`, |coef| `0.001950`
- `lag_00__CT_kills_last_3s`: coefficient `0.001597`, |coef| `0.001597`
- `lag_15__T_place_LOWERMID`: coefficient `0.001455`, |coef| `0.001455`
- `lag_00__damage_diff_last_5s`: coefficient `0.001453`, |coef| `0.001453`
- `lag_08__CT2__is_walking`: coefficient `0.001383`, |coef| `0.001383`
- `lag_15__CT5__is_walking`: coefficient `0.001376`, |coef| `0.001376`
- `lag_14__T3__duck_amount`: coefficient `-0.001352`, |coef| `0.001352`
- `lag_13__CT_place_TOPOFMID`: coefficient `0.001339`, |coef| `0.001339`
- `lag_13__T_place_LOWERMID`: coefficient `-0.001326`, |coef| `0.001326`
- `lag_06__CT2__is_walking`: coefficient `-0.001314`, |coef| `0.001314`
- `lag_03__T_flashed_players`: coefficient `0.001300`, |coef| `0.001300`
- `lag_01__CT2__shots_fired`: coefficient `0.001290`, |coef| `0.001290`
- `lag_15__T3__duck_amount`: coefficient `-0.001289`, |coef| `0.001289`

## Top 10 utility ridge features

- `lag_00__T4__flash`: coefficient `-0.000945` (lowers CT win probability)
- `lag_03__T2__flash_duration`: coefficient `0.000934` (raises CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000835` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000812` (lowers CT win probability)
- `lag_14__T2__smoke`: coefficient `0.000807` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000757` (lowers CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000735` (raises CT win probability)
- `lag_02__T2__flash_duration`: coefficient `0.000724` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000712` (raises CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000670` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002305` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002175` (raises CT win probability)
- `lag_01__T_place_ARCH`: coefficient `0.001950` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001597` (raises CT win probability)
- `lag_15__T_place_LOWERMID`: coefficient `0.001455` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001453` (raises CT win probability)
- `lag_08__CT2__is_walking`: coefficient `0.001383` (raises CT win probability)
- `lag_15__CT5__is_walking`: coefficient `0.001376` (raises CT win probability)
- `lag_14__T3__duck_amount`: coefficient `-0.001352` (lowers CT win probability)
- `lag_13__CT_place_TOPOFMID`: coefficient `0.001339` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `6963`, seconds `13.00`, LSTM delta `+0.2191`

Top all feature movements:
- `lag_13__T_place_LOWERMID`: contribution `+0.008824`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008008`
- `lag_03__T_flashed_players`: contribution `+0.007526`
- `lag_00__kill_diff_last_3s`: contribution `+0.005236`
- `lag_13__CT_place_TOPOFMID`: contribution `+0.004857`

Top utility-only movements:
- `lag_03__T2__flash_duration`: contribution `+0.003423`

### tick `10995`, seconds `76.00`, LSTM delta `+0.2149`

Top all feature movements:
- `lag_01__T_place_ARCH`: contribution `+0.018146`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011212`
- `lag_00__kill_diff_last_3s`: contribution `+0.005236`
- `lag_14__T3__duck_amount`: contribution `+0.005098`
- `lag_00__CT_kills_last_3s`: contribution `+0.004611`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10931`, seconds `75.00`, LSTM delta `-0.1574`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.011212`
- `lag_00__kill_diff_last_3s`: contribution `-0.005236`
- `lag_15__T3__duck_amount`: contribution `-0.004861`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004825`
- `lag_01__CT4__shots_fired`: contribution `-0.004582`

Top utility-only movements:
- `lag_14__T2__smoke`: contribution `-0.001772`

### tick `11251`, seconds `80.00`, LSTM delta `+0.1017`

Top all feature movements:
- `lag_02__CT_place_GRAVEYARD`: contribution `+0.020882`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011212`
- `lag_00__kill_diff_last_3s`: contribution `+0.005236`
- `lag_00__CT_kills_last_3s`: contribution `+0.004611`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003447`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `6931`, seconds `12.50`, LSTM delta `+0.0896`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.008008`
- `lag_12__T_place_LOWERMID`: contribution `+0.006080`
- `lag_02__T_flashed_players`: contribution `+0.005308`
- `lag_13__CT_place_TOPOFMID`: contribution `+0.004857`
- `lag_15__T_place_LOWERMID`: contribution `+0.004841`

Top utility-only movements:
- `lag_02__T2__flash_duration`: contribution `+0.002654`
- `lag_02__T_flash_duration_sum`: contribution `+0.001675`
