# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `3`

## Largest probability jumps

- tick `26904`, seconds `73.00`, LSTM `0.7878`, delta `+0.1839`
- tick `23736`, seconds `23.50`, LSTM `0.5500`, delta `+0.1450`
- tick `28920`, seconds `104.50`, LSTM `0.9456`, delta `+0.1445`
- tick `23320`, seconds `17.00`, LSTM `0.4752`, delta `-0.1358`
- tick `27000`, seconds `74.50`, LSTM `0.9452`, delta `+0.1080`
- tick `28696`, seconds `101.00`, LSTM `0.8408`, delta `+0.1050`
- tick `24024`, seconds `28.00`, LSTM `0.6221`, delta `+0.0702`
- tick `28888`, seconds `104.00`, LSTM `0.8011`, delta `-0.0660`
- tick `27096`, seconds `76.00`, LSTM `0.8320`, delta `-0.0580`
- tick `28632`, seconds `100.00`, LSTM `0.6849`, delta `-0.0530`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003655`, |coef| `0.003655`
- `lag_00__kill_diff_last_3s`: coefficient `0.003393`, |coef| `0.003393`
- `lag_00__damage_diff_last_5s`: coefficient `0.002276`, |coef| `0.002276`
- `lag_00__T2__duck_amount`: coefficient `-0.002091`, |coef| `0.002091`
- `lag_00__CT_damage_last_5s`: coefficient `0.002036`, |coef| `0.002036`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002035`, |coef| `0.002035`
- `lag_00__T3__is_walking`: coefficient `-0.001907`, |coef| `0.001907`
- `lag_00__T5__alive`: coefficient `-0.001813`, |coef| `0.001813`
- `lag_02__T2__duck_amount`: coefficient `0.001763`, |coef| `0.001763`
- `lag_10__CT_place_HOUSE`: coefficient `-0.001756`, |coef| `0.001756`
- `lag_00__CT2__duck_amount`: coefficient `-0.001745`, |coef| `0.001745`
- `lag_10__CT2__duck_amount`: coefficient `0.001734`, |coef| `0.001734`
- `lag_09__T_place_TSIDEUPPER`: coefficient `-0.001733`, |coef| `0.001733`
- `lag_00__T_place_MAINHALL`: coefficient `-0.001626`, |coef| `0.001626`
- `lag_00__T5__smoke`: coefficient `-0.001608`, |coef| `0.001608`

## Top 10 utility ridge features

- `lag_00__T5__smoke`: coefficient `-0.001608` (lowers CT win probability)
- `lag_00__T5__utility_total`: coefficient `-0.001183` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.001168` (raises CT win probability)
- `lag_05__T1__flash_duration`: coefficient `-0.001078` (lowers CT win probability)
- `lag_13__T_B_site_active_smokes`: coefficient `0.001035` (raises CT win probability)
- `lag_00__T5__flash`: coefficient `-0.001016` (lowers CT win probability)
- `lag_05__T3__flash_duration`: coefficient `-0.000980` (lowers CT win probability)
- `lag_12__T2__flash_duration`: coefficient `-0.000959` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.000901` (lowers CT win probability)
- `lag_09__utility_damage_diff_last_5s`: coefficient `-0.000882` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003655` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003393` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002276` (raises CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.002091` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002036` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002035` (raises CT win probability)
- `lag_00__T3__is_walking`: coefficient `-0.001907` (lowers CT win probability)
- `lag_00__T5__alive`: coefficient `-0.001813` (lowers CT win probability)
- `lag_02__T2__duck_amount`: coefficient `0.001763` (raises CT win probability)
- `lag_10__CT_place_HOUSE`: coefficient `-0.001756` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `26904`, seconds `73.00`, LSTM delta `+0.1839`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010552`
- `lag_00__kill_diff_last_3s`: contribution `+0.008166`
- `lag_10__CT2__duck_amount`: contribution `+0.006605`
- `lag_10__CT_place_HOUSE`: contribution `+0.006203`
- `lag_14__T5__duck_amount`: contribution `+0.005894`

Top utility-only movements:
- `lag_00__T5__smoke`: contribution `+0.003484`

### tick `23736`, seconds `23.50`, LSTM delta `+0.1450`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010552`
- `lag_00__kill_diff_last_3s`: contribution `+0.008166`
- `lag_13__CT_place_TSIDEUPPER`: contribution `+0.006691`
- `lag_05__T1__flash_duration`: contribution `+0.006472`
- `lag_05__T3__flash_duration`: contribution `+0.005846`

Top utility-only movements:
- `lag_05__T1__flash_duration`: contribution `+0.006472`
- `lag_05__T3__flash_duration`: contribution `+0.005846`
- `lag_12__T2__flash_duration`: contribution `+0.004628`
- `lag_06__T5__flash_duration`: contribution `+0.004108`
- `lag_05__CT1__flash_duration`: contribution `+0.003870`

### tick `28920`, seconds `104.50`, LSTM delta `+0.1445`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010552`
- `lag_00__T_duck_amount_mean`: contribution `+0.008786`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008482`
- `lag_00__kill_diff_last_3s`: contribution `+0.008166`
- `lag_00__T_place_SIDEHALL`: contribution `+0.006484`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.005111`
- `lag_07__T3__flash_duration`: contribution `+0.004983`
- `lag_09__T_utility_damage_last_5s`: contribution `+0.004316`
- `lag_09__utility_damage_diff_last_5s`: contribution `+0.003583`
- `lag_03__CT5__flash_duration`: contribution `+0.003254`

### tick `23320`, seconds `17.00`, LSTM delta `-0.1358`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.009120`
- `lag_00__kill_diff_last_3s`: contribution `-0.008166`
- `lag_03__CT_place_TSIDEUPPER`: contribution `-0.006937`
- `lag_00__T2__duck_amount`: contribution `-0.005504`
- `lag_11__CT1__flash_duration`: contribution `-0.005133`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `-0.005133`
- `lag_03__T3__flash_duration`: contribution `-0.004966`
- `lag_04__T1__flash_duration`: contribution `-0.002773`
- `lag_04__T5__flash_duration`: contribution `-0.002672`

### tick `27000`, seconds `74.50`, LSTM delta `+0.1080`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.010552`
- `lag_00__kill_diff_last_3s`: contribution `+0.008166`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007069`
- `lag_00__T_place_MAINHALL`: contribution `+0.005870`
- `lag_00__damage_diff_last_5s`: contribution `+0.005134`

Top utility-only movements:
- No utility movement among the top local contributors.
