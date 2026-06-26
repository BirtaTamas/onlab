# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-gamerlegion-vs-tyloo-bo3-CHuj0-KFwAe9c3Zh96vlUq/gamerlegion-vs-tyloo-m2-ancient.csv`
- round_num: `12`

## Largest probability jumps

- tick `134762`, seconds `24.50`, LSTM `0.8487`, delta `+0.2249`
- tick `136650`, seconds `54.00`, LSTM `0.9645`, delta `+0.0525`
- tick `136202`, seconds `47.00`, LSTM `0.9108`, delta `+0.0263`
- tick `135402`, seconds `34.50`, LSTM `0.8607`, delta `-0.0246`
- tick `134538`, seconds `21.00`, LSTM `0.5765`, delta `-0.0239`
- tick `136106`, seconds `45.50`, LSTM `0.9017`, delta `+0.0239`
- tick `136170`, seconds `46.50`, LSTM `0.8844`, delta `-0.0209`
- tick `134474`, seconds `20.00`, LSTM `0.6020`, delta `+0.0204`
- tick `134154`, seconds `15.00`, LSTM `0.5973`, delta `-0.0204`
- tick `135338`, seconds `33.50`, LSTM `0.8811`, delta `+0.0196`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001955`, |coef| `0.001955`
- `lag_00__kill_diff_last_3s`: coefficient `0.001630`, |coef| `0.001630`
- `lag_00__damage_diff_last_5s`: coefficient `0.001558`, |coef| `0.001558`
- `lag_00__CT_damage_last_5s`: coefficient `0.001501`, |coef| `0.001501`
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.001479`, |coef| `0.001479`
- `lag_08__T3__flash_duration`: coefficient `-0.001414`, |coef| `0.001414`
- `lag_09__T4__flash_duration`: coefficient `-0.001325`, |coef| `0.001325`
- `lag_02__CT3__is_scoped`: coefficient `0.001111`, |coef| `0.001111`
- `lag_00__T_place_MIDDLE`: coefficient `-0.001037`, |coef| `0.001037`
- `lag_00__T4__flash`: coefficient `-0.000980`, |coef| `0.000980`
- `lag_02__CT3__is_walking`: coefficient `-0.000949`, |coef| `0.000949`
- `lag_12__T2__flash_duration`: coefficient `-0.000949`, |coef| `0.000949`
- `lag_09__T_flashed_players`: coefficient `-0.000941`, |coef| `0.000941`
- `lag_00__T4__alive`: coefficient `-0.000933`, |coef| `0.000933`
- `lag_10__T_place_TSIDELOWER`: coefficient `-0.000931`, |coef| `0.000931`

## Top 10 utility ridge features

- `lag_08__T3__flash_duration`: coefficient `-0.001414` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.001325` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000980` (lowers CT win probability)
- `lag_12__T2__flash_duration`: coefficient `-0.000949` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000891` (lowers CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `-0.000858` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.000773` (lowers CT win probability)
- `lag_02__CT4__smoke`: coefficient `-0.000770` (lowers CT win probability)
- `lag_06__T2__smoke`: coefficient `-0.000753` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000721` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001955` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001630` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001558` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001501` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `-0.001479` (lowers CT win probability)
- `lag_02__CT3__is_scoped`: coefficient `0.001111` (raises CT win probability)
- `lag_00__T_place_MIDDLE`: coefficient `-0.001037` (lowers CT win probability)
- `lag_02__CT3__is_walking`: coefficient `-0.000949` (lowers CT win probability)
- `lag_09__T_flashed_players`: coefficient `-0.000941` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.000933` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `134762`, seconds `24.50`, LSTM delta `+0.2249`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011287`
- `lag_00__kill_diff_last_3s`: contribution `+0.007845`
- `lag_08__T3__flash_duration`: contribution `+0.007607`
- `lag_00__damage_diff_last_5s`: contribution `+0.007028`
- `lag_09__T4__flash_duration`: contribution `+0.006981`

Top utility-only movements:
- `lag_08__T3__flash_duration`: contribution `+0.007607`
- `lag_09__T4__flash_duration`: contribution `+0.006981`
- `lag_09__T_flash_duration_sum`: contribution `+0.003612`
- `lag_09__T1__flash_duration`: contribution `+0.003551`
- `lag_12__T2__flash_duration`: contribution `+0.002859`

### tick `136650`, seconds `54.00`, LSTM delta `+0.0525`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.005644`
- `lag_05__T_place_SIDEENTRANCE`: contribution `+0.004375`
- `lag_00__kill_diff_last_3s`: contribution `+0.003922`
- `lag_00__damage_diff_last_5s`: contribution `+0.002425`
- `lag_00__CT_damage_last_5s`: contribution `+0.002257`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `136202`, seconds `47.00`, LSTM delta `+0.0263`

Top all feature movements:
- `lag_02__CT3__is_scoped`: contribution `+0.005053`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `+0.002570`
- `lag_02__CT3__is_walking`: contribution `+0.002265`
- `lag_00__T3__is_walking`: contribution `+0.002118`
- `lag_06__CT2__is_walking`: contribution `-0.001240`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `135402`, seconds `34.50`, LSTM delta `-0.0246`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.002550`
- `lag_07__CT_place_SIDEHALL`: contribution `-0.002254`
- `lag_00__T3__is_walking`: contribution `-0.002118`
- `lag_00__CT2__duck_amount`: contribution `-0.001720`
- `lag_08__CT5__is_walking`: contribution `+0.001526`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.001294`

### tick `134538`, seconds `21.00`, LSTM delta `-0.0239`

Top all feature movements:
- `lag_14__T_flashed_players`: contribution `-0.002630`
- `lag_00__T3__is_walking`: contribution `-0.002118`
- `lag_00__CT_shots_fired_sum`: contribution `-0.001700`
- `lag_00__T_place_MIDDLE`: contribution `-0.001685`
- `lag_05__T2__is_walking`: contribution `-0.001559`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.001346`
