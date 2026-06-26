# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `65689`, seconds `20.00`, LSTM `0.1443`, delta `-0.1978`
- tick `67289`, seconds `45.00`, LSTM `0.3756`, delta `+0.1935`
- tick `67353`, seconds `46.00`, LSTM `0.1467`, delta `-0.1797`
- tick `65881`, seconds `23.00`, LSTM `0.2717`, delta `+0.1037`
- tick `65337`, seconds `14.50`, LSTM `0.4194`, delta `+0.0907`
- tick `67513`, seconds `48.50`, LSTM `0.0356`, delta `-0.0636`
- tick `65945`, seconds `24.00`, LSTM `0.3463`, delta `+0.0581`
- tick `66073`, seconds `26.00`, LSTM `0.2749`, delta `-0.0516`
- tick `67321`, seconds `45.50`, LSTM `0.3264`, delta `-0.0492`
- tick `65273`, seconds `13.50`, LSTM `0.3680`, delta `-0.0474`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002331`, |coef| `0.002331`
- `lag_08__CT4__flash_duration`: coefficient `0.002212`, |coef| `0.002212`
- `lag_08__T_flashed_players`: coefficient `0.001989`, |coef| `0.001989`
- `lag_12__T2__duck_amount`: coefficient `0.001944`, |coef| `0.001944`
- `lag_00__T3__shots_fired`: coefficient `-0.001930`, |coef| `0.001930`
- `lag_00__T_kills_last_3s`: coefficient `-0.001825`, |coef| `0.001825`
- `lag_01__CT_place_LIBRARY`: coefficient `-0.001819`, |coef| `0.001819`
- `lag_10__CT4__flash_duration`: coefficient `-0.001777`, |coef| `0.001777`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001659`, |coef| `0.001659`
- `lag_11__T3__shots_fired`: coefficient `0.001497`, |coef| `0.001497`
- `lag_11__T_shots_fired_sum`: coefficient `0.001376`, |coef| `0.001376`
- `lag_07__CT_place_RUINS`: coefficient `-0.001376`, |coef| `0.001376`
- `lag_11__T2__duck_amount`: coefficient `-0.001344`, |coef| `0.001344`
- `lag_06__CT_place_LIBRARY`: coefficient `0.001342`, |coef| `0.001342`
- `lag_03__CT_place_LIBRARY`: coefficient `0.001337`, |coef| `0.001337`

## Top 10 utility ridge features

- `lag_08__CT4__flash_duration`: coefficient `0.002212` (raises CT win probability)
- `lag_10__CT4__flash_duration`: coefficient `-0.001777` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001117` (raises CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001037` (lowers CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000995` (lowers CT win probability)
- `lag_10__CT_flash_duration_sum`: coefficient `-0.000986` (lowers CT win probability)
- `lag_06__T1__smoke`: coefficient `0.000938` (raises CT win probability)
- `lag_08__CT_flash_duration_sum`: coefficient `0.000889` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `-0.000831` (lowers CT win probability)
- `lag_04__T1__smoke`: coefficient `-0.000805` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002331` (raises CT win probability)
- `lag_08__T_flashed_players`: coefficient `0.001989` (raises CT win probability)
- `lag_12__T2__duck_amount`: coefficient `0.001944` (raises CT win probability)
- `lag_00__T3__shots_fired`: coefficient `-0.001930` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001825` (lowers CT win probability)
- `lag_01__CT_place_LIBRARY`: coefficient `-0.001819` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001659` (lowers CT win probability)
- `lag_11__T3__shots_fired`: coefficient `0.001497` (raises CT win probability)
- `lag_11__T_shots_fired_sum`: coefficient `0.001376` (raises CT win probability)
- `lag_07__CT_place_RUINS`: coefficient `-0.001376` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65689`, seconds `20.00`, LSTM delta `-0.1978`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `-0.024764`
- `lag_11__T3__shots_fired`: contribution `-0.021752`
- `lag_13__CT_place_BALCONY`: contribution `-0.007292`
- `lag_00__T_kills_last_3s`: contribution `-0.005783`
- `lag_00__kill_diff_last_3s`: contribution `-0.005610`

Top utility-only movements:
- `lag_04__T_utility_damage_last_5s`: contribution `-0.004783`
- `lag_01__T1__flash_duration`: contribution `-0.004430`
- `lag_12__CT5__flash_duration`: contribution `-0.004146`
- `lag_12__T3__flash_duration`: contribution `-0.003123`
- `lag_05__CT_B_site_active_infernos`: contribution `-0.002748`

### tick `67289`, seconds `45.00`, LSTM delta `+0.1935`

Top all feature movements:
- `lag_08__CT4__flash_duration`: contribution `+0.014666`
- `lag_01__CT_place_LIBRARY`: contribution `+0.011662`
- `lag_06__CT_place_LIBRARY`: contribution `+0.008608`
- `lag_08__T_flashed_players`: contribution `+0.007676`
- `lag_12__T2__duck_amount`: contribution `+0.007434`

Top utility-only movements:
- `lag_08__CT4__flash_duration`: contribution `+0.014666`
- `lag_08__CT_flash_duration_sum`: contribution `+0.002636`

### tick `67353`, seconds `46.00`, LSTM delta `-0.1797`

Top all feature movements:
- `lag_10__CT4__flash_duration`: contribution `-0.011785`
- `lag_03__CT_place_LIBRARY`: contribution `-0.008575`
- `lag_00__CT4__flash_duration`: contribution `-0.007404`
- `lag_00__T_kills_last_3s`: contribution `-0.005783`
- `lag_00__kill_diff_last_3s`: contribution `-0.005610`

Top utility-only movements:
- `lag_10__CT4__flash_duration`: contribution `-0.011785`
- `lag_00__CT4__flash_duration`: contribution `-0.007404`
- `lag_10__CT_flash_duration_sum`: contribution `-0.002925`

### tick `65881`, seconds `23.00`, LSTM delta `+0.1037`

Top all feature movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.007399`
- `lag_00__T_shots_fired_sum`: contribution `+0.006218`
- `lag_00__T3__shots_fired`: contribution `+0.005842`
- `lag_00__T_kills_last_3s`: contribution `+0.005783`
- `lag_00__kill_diff_last_3s`: contribution `+0.005610`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `+0.007399`
- `lag_03__CT_utility_damage_last_5s`: contribution `+0.004713`
- `lag_03__utility_damage_diff_last_5s`: contribution `+0.003045`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.002730`
- `lag_11__CT_B_site_active_infernos`: contribution `+0.002395`

### tick `65337`, seconds `14.50`, LSTM delta `+0.0907`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.029848`
- `lag_00__T3__shots_fired`: contribution `+0.028044`
- `lag_02__CT_place_BALCONY`: contribution `+0.004105`
- `lag_07__CT4__is_scoped`: contribution `-0.003644`
- `lag_03__CT4__is_walking`: contribution `+0.002796`

Top utility-only movements:
- `lag_05__CT_B_site_active_infernos`: contribution `+0.002748`
- `lag_05__T4__flash_duration`: contribution `+0.002582`
- `lag_03__T_B_site_active_infernos`: contribution `+0.002148`
- `lag_01__CT5__flash_duration`: contribution `+0.001799`
