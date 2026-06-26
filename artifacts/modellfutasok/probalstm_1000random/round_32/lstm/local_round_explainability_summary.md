# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `18`

## Largest probability jumps

- tick `118510`, seconds `14.00`, LSTM `0.1909`, delta `-0.2963`
- tick `119790`, seconds `34.00`, LSTM `0.6923`, delta `+0.2379`
- tick `119918`, seconds `36.00`, LSTM `0.8631`, delta `+0.1949`
- tick `118446`, seconds `13.00`, LSTM `0.5339`, delta `-0.1335`
- tick `119022`, seconds `22.00`, LSTM `0.3174`, delta `+0.1056`
- tick `119566`, seconds `30.50`, LSTM `0.2588`, delta `+0.0905`
- tick `118542`, seconds `14.50`, LSTM `0.1184`, delta `-0.0725`
- tick `118926`, seconds `20.50`, LSTM `0.2558`, delta `+0.0644`
- tick `118350`, seconds `11.50`, LSTM `0.6602`, delta `-0.0617`
- tick `119726`, seconds `33.00`, LSTM `0.3972`, delta `+0.0588`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001955`, |coef| `0.001955`
- `lag_15__T4__flash_duration`: coefficient `-0.001743`, |coef| `0.001743`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001639`, |coef| `0.001639`
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.001635`, |coef| `0.001635`
- `lag_07__T2__flash_duration`: coefficient `0.001617`, |coef| `0.001617`
- `lag_00__CT_kills_last_3s`: coefficient `0.001614`, |coef| `0.001614`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.001549`, |coef| `0.001549`
- `lag_08__T4__flash_duration`: coefficient `-0.001468`, |coef| `0.001468`
- `lag_14__T4__flash_duration`: coefficient `-0.001451`, |coef| `0.001451`
- `lag_00__damage_diff_last_5s`: coefficient `0.001408`, |coef| `0.001408`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001366`, |coef| `0.001366`
- `lag_00__T_macro_B`: coefficient `-0.001366`, |coef| `0.001366`
- `lag_04__T1__flash_duration`: coefficient `-0.001349`, |coef| `0.001349`
- `lag_06__CT_flashed_players`: coefficient `-0.001283`, |coef| `0.001283`
- `lag_06__T2__flash_duration`: coefficient `0.001275`, |coef| `0.001275`

## Top 10 utility ridge features

- `lag_15__T4__flash_duration`: coefficient `-0.001743` (lowers CT win probability)
- `lag_07__T2__flash_duration`: coefficient `0.001617` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `-0.001468` (lowers CT win probability)
- `lag_14__T4__flash_duration`: coefficient `-0.001451` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `-0.001349` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.001275` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `-0.001275` (lowers CT win probability)
- `lag_13__T4__flash_duration`: coefficient `-0.001229` (lowers CT win probability)
- `lag_08__CT_active_smokes`: coefficient `-0.001158` (lowers CT win probability)
- `lag_11__T2__flash_duration`: coefficient `0.001154` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001955` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001639` (raises CT win probability)
- `lag_13__CT_place_TSIDEUPPER`: coefficient `-0.001635` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001614` (raises CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.001549` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001408` (raises CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001366` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001366` (lowers CT win probability)
- `lag_06__CT_flashed_players`: coefficient `-0.001283` (lowers CT win probability)
- `lag_05__CT_shots_fired_sum`: coefficient `-0.001273` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `118510`, seconds `14.00`, LSTM delta `-0.2963`

Top all feature movements:
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.011646`
- `lag_04__T1__flash_duration`: contribution `-0.009820`
- `lag_06__CT_flashed_players`: contribution `-0.008428`
- `lag_07__T_place_TSIDELOWER`: contribution `-0.008284`
- `lag_14__CT_place_HOUSE`: contribution `-0.008020`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `-0.009820`
- `lag_06__CT5__flash_duration`: contribution `-0.005267`
- `lag_02__CT5__flash_duration`: contribution `-0.004769`

### tick `119790`, seconds `34.00`, LSTM delta `+0.2379`

Top all feature movements:
- `lag_15__T4__flash_duration`: contribution `+0.011957`
- `lag_07__T2__flash_duration`: contribution `+0.010624`
- `lag_02__CT_shots_fired_sum`: contribution `+0.010251`
- `lag_01__CT_shots_fired_sum`: contribution `+0.005850`
- `lag_00__kill_diff_last_3s`: contribution `+0.004706`

Top utility-only movements:
- `lag_15__T4__flash_duration`: contribution `+0.011957`
- `lag_07__T2__flash_duration`: contribution `+0.010624`
- `lag_01__CT_B_site_active_infernos`: contribution `+0.004381`
- `lag_07__T_flash_duration_sum`: contribution `+0.003369`

### tick `119918`, seconds `36.00`, LSTM delta `+0.1949`

Top all feature movements:
- `lag_05__CT_shots_fired_sum`: contribution `+0.013268`
- `lag_11__T2__flash_duration`: contribution `+0.007580`
- `lag_06__CT_shots_fired_sum`: contribution `+0.005840`
- `lag_05__CT2__shots_fired`: contribution `+0.004914`
- `lag_00__kill_diff_last_3s`: contribution `+0.004706`

Top utility-only movements:
- `lag_11__T2__flash_duration`: contribution `+0.007580`

### tick `118446`, seconds `13.00`, LSTM delta `-0.1335`

Top all feature movements:
- `lag_14__T_place_WATER`: contribution `-0.010562`
- `lag_15__T_place_TUNNEL`: contribution `-0.007410`
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.005278`
- `lag_00__kill_diff_last_3s`: contribution `-0.004706`
- `lag_13__T_place_TUNNEL`: contribution `-0.004692`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.004172`
- `lag_00__CT5__flash_duration`: contribution `-0.002110`

### tick `119022`, seconds `22.00`, LSTM delta `+0.1056`

Top all feature movements:
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.007075`
- `lag_04__T4__flash_duration`: contribution `+0.004949`
- `lag_00__kill_diff_last_3s`: contribution `+0.004706`
- `lag_00__CT_kills_last_3s`: contribution `+0.004660`
- `lag_06__T_place_RAMP`: contribution `-0.004025`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.004949`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.004007`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.003931`
- `lag_06__T1__flash_duration`: contribution `+0.003288`
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.002436`
