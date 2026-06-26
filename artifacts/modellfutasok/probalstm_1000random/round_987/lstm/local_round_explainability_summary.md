# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `6`

## Largest probability jumps

- tick `48179`, seconds `94.00`, LSTM `0.9099`, delta `+0.2233`
- tick `47219`, seconds `79.00`, LSTM `0.8272`, delta `+0.1705`
- tick `48371`, seconds `97.00`, LSTM `0.7893`, delta `-0.1251`
- tick `44691`, seconds `39.50`, LSTM `0.6843`, delta `-0.1109`
- tick `43731`, seconds `24.50`, LSTM `0.7024`, delta `+0.1095`
- tick `47347`, seconds `81.00`, LSTM `0.9446`, delta `+0.0842`
- tick `47539`, seconds `84.00`, LSTM `0.7624`, delta `-0.0690`
- tick `47443`, seconds `82.50`, LSTM `0.8975`, delta `-0.0501`
- tick `47731`, seconds `87.00`, LSTM `0.8105`, delta `+0.0438`
- tick `48147`, seconds `93.50`, LSTM `0.6866`, delta `-0.0413`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004073`, |coef| `0.004073`
- `lag_00__CT_kills_last_3s`: coefficient `0.003871`, |coef| `0.003871`
- `lag_01__T1__flash_duration`: coefficient `-0.003413`, |coef| `0.003413`
- `lag_14__T1__flash_duration`: coefficient `0.003121`, |coef| `0.003121`
- `lag_00__damage_diff_last_5s`: coefficient `0.002460`, |coef| `0.002460`
- `lag_10__T3__flash_duration`: coefficient `-0.002168`, |coef| `0.002168`
- `lag_00__CT_damage_last_5s`: coefficient `0.001859`, |coef| `0.001859`
- `lag_08__T_place_UNDERPASS`: coefficient `0.001858`, |coef| `0.001858`
- `lag_01__T_flash_duration_sum`: coefficient `-0.001695`, |coef| `0.001695`
- `lag_08__CT3__duck_amount`: coefficient `0.001660`, |coef| `0.001660`
- `lag_11__CT3__is_walking`: coefficient `-0.001463`, |coef| `0.001463`
- `lag_15__T_place_BALCONY`: coefficient `-0.001450`, |coef| `0.001450`
- `lag_00__CT_duck_amount_mean`: coefficient `0.001448`, |coef| `0.001448`
- `lag_02__T2__duck_amount`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_00__CT_walking_count`: coefficient `-0.001429`, |coef| `0.001429`

## Top 10 utility ridge features

- `lag_01__T1__flash_duration`: coefficient `-0.003413` (lowers CT win probability)
- `lag_14__T1__flash_duration`: coefficient `0.003121` (raises CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.002168` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `-0.001695` (lowers CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `0.001359` (raises CT win probability)
- `lag_06__T1__flash_duration`: coefficient `0.001054` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001034` (raises CT win probability)
- `lag_07__T1__flash_duration`: coefficient `0.001011` (raises CT win probability)
- `lag_04__T3__smoke`: coefficient `-0.001002` (lowers CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.000968` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004073` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003871` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002460` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001859` (raises CT win probability)
- `lag_08__T_place_UNDERPASS`: coefficient `0.001858` (raises CT win probability)
- `lag_08__CT3__duck_amount`: coefficient `0.001660` (raises CT win probability)
- `lag_11__CT3__is_walking`: coefficient `-0.001463` (lowers CT win probability)
- `lag_15__T_place_BALCONY`: coefficient `-0.001450` (lowers CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.001448` (raises CT win probability)
- `lag_02__T2__duck_amount`: coefficient `-0.001436` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `48179`, seconds `94.00`, LSTM delta `+0.2233`

Top all feature movements:
- `lag_01__T1__flash_duration`: contribution `+0.024713`
- `lag_14__T1__flash_duration`: contribution `+0.019237`
- `lag_10__T3__flash_duration`: contribution `+0.011737`
- `lag_00__CT_kills_last_3s`: contribution `+0.011177`
- `lag_00__kill_diff_last_3s`: contribution `+0.009802`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `+0.024713`
- `lag_14__T1__flash_duration`: contribution `+0.019237`
- `lag_10__T3__flash_duration`: contribution `+0.011737`
- `lag_01__T_flash_duration_sum`: contribution `+0.005105`
- `lag_14__T_flash_duration_sum`: contribution `+0.003487`

### tick `47219`, seconds `79.00`, LSTM delta `+0.1705`

Top all feature movements:
- `lag_15__T_place_BALCONY`: contribution `+0.019943`
- `lag_00__CT_kills_last_3s`: contribution `+0.011177`
- `lag_00__kill_diff_last_3s`: contribution `+0.009802`
- `lag_08__T_place_UNDERPASS`: contribution `+0.007279`
- `lag_02__T2__duck_amount`: contribution `+0.005492`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `48371`, seconds `97.00`, LSTM delta `-0.1251`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.019605`
- `lag_00__CT_kills_last_3s`: contribution `-0.011177`
- `lag_00__T_shots_fired_sum`: contribution `-0.008607`
- `lag_07__T1__flash_duration`: contribution `-0.007323`
- `lag_04__CT_place_LIBRARY`: contribution `-0.007293`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `-0.007323`
- `lag_07__T_flash_duration_sum`: contribution `-0.001864`

### tick `44691`, seconds `39.50`, LSTM delta `-0.1109`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009802`
- `lag_01__T_place_BALCONY`: contribution `-0.009035`
- `lag_00__CT4__flash_duration`: contribution `-0.008179`
- `lag_00__damage_diff_last_5s`: contribution `-0.005550`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.004516`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.008179`
- `lag_05__CT_utility_damage_last_5s`: contribution `-0.004516`
- `lag_15__T5__flash_duration`: contribution `-0.003925`
- `lag_14__CT4__flash_duration`: contribution `-0.003751`
- `lag_15__CT3__flash_duration`: contribution `-0.003627`

### tick `43731`, seconds `24.50`, LSTM delta `+0.1095`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011177`
- `lag_00__kill_diff_last_3s`: contribution `+0.009802`
- `lag_02__CT1__flash_duration`: contribution `+0.006706`
- `lag_02__CT_place_LIBRARY`: contribution `+0.005805`
- `lag_02__T2__duck_amount`: contribution `+0.005492`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.006706`
- `lag_02__CT_flash_duration_sum`: contribution `+0.002830`
