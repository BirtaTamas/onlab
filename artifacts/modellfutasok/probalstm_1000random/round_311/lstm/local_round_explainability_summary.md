# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `40399`, seconds `13.00`, LSTM `0.9068`, delta `+0.1423`
- tick `40367`, seconds `12.50`, LSTM `0.7644`, delta `+0.1279`
- tick `40815`, seconds `19.50`, LSTM `0.9581`, delta `+0.0626`
- tick `44975`, seconds `84.50`, LSTM `0.9287`, delta `-0.0372`
- tick `43951`, seconds `68.50`, LSTM `0.9720`, delta `+0.0330`
- tick `40783`, seconds `19.00`, LSTM `0.8954`, delta `-0.0306`
- tick `40239`, seconds `10.50`, LSTM `0.6288`, delta `+0.0251`
- tick `40079`, seconds `8.00`, LSTM `0.6013`, delta `+0.0240`
- tick `40175`, seconds `9.50`, LSTM `0.6205`, delta `+0.0182`
- tick `40431`, seconds `13.50`, LSTM `0.9235`, delta `+0.0168`

## Top 15 local ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001285`, |coef| `0.001285`
- `lag_03__T4__shots_fired`: coefficient `0.000945`, |coef| `0.000945`
- `lag_02__T4__shots_fired`: coefficient `0.000925`, |coef| `0.000925`
- `lag_02__T_shots_fired_sum`: coefficient `0.000901`, |coef| `0.000901`
- `lag_02__T5__shots_fired`: coefficient `0.000901`, |coef| `0.000901`
- `lag_01__T5__shots_fired`: coefficient `0.000855`, |coef| `0.000855`
- `lag_04__T4__shots_fired`: coefficient `0.000829`, |coef| `0.000829`
- `lag_01__T4__shots_fired`: coefficient `0.000786`, |coef| `0.000786`
- `lag_00__kill_diff_last_3s`: coefficient `0.000734`, |coef| `0.000734`
- `lag_00__T5__shots_fired`: coefficient `0.000712`, |coef| `0.000712`
- `lag_03__T_shots_fired_sum`: coefficient `0.000698`, |coef| `0.000698`
- `lag_03__T5__shots_fired`: coefficient `0.000673`, |coef| `0.000673`
- `lag_00__T4__shots_fired`: coefficient `0.000655`, |coef| `0.000655`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000635`, |coef| `0.000635`
- `lag_00__CT_kills_last_3s`: coefficient `0.000617`, |coef| `0.000617`

## Top 10 utility ridge features

- `lag_05__T5__flash_duration`: coefficient `0.000558` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `0.000540` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `0.000460` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000455` (raises CT win probability)
- `lag_04__T2__flash_duration`: coefficient `0.000451` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `0.000443` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000417` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000408` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.000378` (lowers CT win probability)
- `lag_02__CT_active_infernos`: coefficient `0.000373` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_shots_fired_sum`: coefficient `-0.001285` (lowers CT win probability)
- `lag_03__T4__shots_fired`: coefficient `0.000945` (raises CT win probability)
- `lag_02__T4__shots_fired`: coefficient `0.000925` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `0.000901` (raises CT win probability)
- `lag_02__T5__shots_fired`: coefficient `0.000901` (raises CT win probability)
- `lag_01__T5__shots_fired`: coefficient `0.000855` (raises CT win probability)
- `lag_04__T4__shots_fired`: coefficient `0.000829` (raises CT win probability)
- `lag_01__T4__shots_fired`: coefficient `0.000786` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000734` (raises CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.000712` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `40399`, seconds `13.00`, LSTM delta `+0.1423`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.014452`
- `lag_02__T_shots_fired_sum`: contribution `+0.006756`
- `lag_03__T_shots_fired_sum`: contribution `+0.005233`
- `lag_13__T_place_LOWERMID`: contribution `+0.003334`
- `lag_03__T4__shots_fired`: contribution `+0.002918`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.002709`
- `lag_03__T1__flash_duration`: contribution `+0.002645`
- `lag_05__T5__flash_duration`: contribution `+0.002615`

### tick `40367`, seconds `12.50`, LSTM delta `+0.1279`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.011561`
- `lag_02__T_shots_fired_sum`: contribution `+0.006756`
- `lag_03__T4__shots_fired`: contribution `+0.002918`
- `lag_02__T4__shots_fired`: contribution `+0.002858`
- `lag_04__T_flashed_players`: contribution `+0.002835`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.002658`
- `lag_02__T1__flash_duration`: contribution `+0.002577`
- `lag_04__T5__flash_duration`: contribution `+0.002530`
- `lag_04__T_flash_duration_sum`: contribution `+0.001895`
- `lag_00__T4__flash_duration`: contribution `+0.001815`

### tick `40815`, seconds `19.50`, LSTM delta `+0.0626`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.005320`
- `lag_14__T_shots_fired_sum`: contribution `+0.003706`
- `lag_13__T_shots_fired_sum`: contribution `+0.002574`
- `lag_00__CT_kills_last_3s`: contribution `+0.001783`
- `lag_03__CT_place_RUINS`: contribution `+0.001767`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `+0.001636`
- `lag_14__T4__flash_duration`: contribution `+0.000936`
- `lag_15__T4__flash_duration`: contribution `+0.000805`

### tick `44975`, seconds `84.50`, LSTM delta `-0.0372`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.001767`
- `lag_00__CT5__is_scoped`: contribution `-0.001308`
- `lag_15__T_duck_amount_mean`: contribution `-0.001146`
- `lag_04__CT5__is_scoped`: contribution `-0.001124`
- `lag_00__T_kills_last_3s`: contribution `-0.000915`

Top utility-only movements:
- `lag_07__T_utility_damage_last_5s`: contribution `-0.000724`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.000662`
- `lag_15__T_A_site_active_infernos`: contribution `-0.000486`

### tick `43951`, seconds `68.50`, LSTM delta `+0.0330`

Top all feature movements:
- `lag_08__CT_place_GRAVEYARD`: contribution `+0.014215`
- `lag_00__CT_kills_last_3s`: contribution `+0.001783`
- `lag_03__CT_place_RUINS`: contribution `+0.001767`
- `lag_00__kill_diff_last_3s`: contribution `+0.001767`
- `lag_07__CT_place_ARCH`: contribution `+0.001350`

Top utility-only movements:
- No utility movement among the top local contributors.
