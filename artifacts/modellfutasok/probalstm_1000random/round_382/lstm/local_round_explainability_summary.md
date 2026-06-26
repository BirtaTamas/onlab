# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-liquid-bo3-pfm398EHUpu3zLY0TgcmxO/the-mongolz-vs-liquid-m2-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `95412`, seconds `36.50`, LSTM `0.5868`, delta `+0.2707`
- tick `95476`, seconds `37.50`, LSTM `0.8564`, delta `+0.2704`
- tick `96372`, seconds `51.50`, LSTM `0.1331`, delta `-0.2042`
- tick `95604`, seconds `39.50`, LSTM `0.7668`, delta `-0.1755`
- tick `95860`, seconds `43.50`, LSTM `0.6090`, delta `-0.1272`
- tick `96084`, seconds `47.00`, LSTM `0.4859`, delta `-0.1099`
- tick `95284`, seconds `34.50`, LSTM `0.3825`, delta `+0.1067`
- tick `96116`, seconds `47.50`, LSTM `0.3988`, delta `-0.0872`
- tick `95348`, seconds `35.50`, LSTM `0.2923`, delta `-0.0679`
- tick `96276`, seconds `50.00`, LSTM `0.3436`, delta `-0.0670`

## Top 15 local ridge features

- `lag_00__CT_place_UNDERPASS`: coefficient `0.002914`, |coef| `0.002914`
- `lag_03__T_bomb_zone_count`: coefficient `-0.002486`, |coef| `0.002486`
- `lag_10__T_place_STAIRS`: coefficient `0.002390`, |coef| `0.002390`
- `lag_08__T_place_STAIRS`: coefficient `0.002181`, |coef| `0.002181`
- `lag_00__kill_diff_last_3s`: coefficient `0.001928`, |coef| `0.001928`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001853`, |coef| `0.001853`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001810`, |coef| `0.001810`
- `lag_10__T_shots_fired_sum`: coefficient `0.001787`, |coef| `0.001787`
- `lag_10__T_place_LADDER`: coefficient `-0.001780`, |coef| `0.001780`
- `lag_00__T2__duck_amount`: coefficient `-0.001703`, |coef| `0.001703`
- `lag_00__T_place_STAIRS`: coefficient `-0.001668`, |coef| `0.001668`
- `lag_04__T_bomb_zone_count`: coefficient `-0.001627`, |coef| `0.001627`
- `lag_00__T_kills_last_3s`: coefficient `-0.001577`, |coef| `0.001577`
- `lag_05__T_place_JUNGLE`: coefficient `-0.001542`, |coef| `0.001542`
- `lag_02__T_place_STAIRS`: coefficient `-0.001540`, |coef| `0.001540`

## Top 10 utility ridge features

- `lag_12__T_A_site_active_infernos`: coefficient `0.001206` (raises CT win probability)
- `lag_10__T2__flash_duration`: coefficient `0.001040` (raises CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `0.000997` (raises CT win probability)
- `lag_11__T_utility_damage_last_5s`: coefficient `0.000823` (raises CT win probability)
- `lag_00__CT_he_last_5s`: coefficient `0.000820` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `0.000816` (raises CT win probability)
- `lag_11__T_B_site_active_smokes`: coefficient `0.000804` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000773` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000756` (raises CT win probability)
- `lag_12__utility_damage_diff_last_5s`: coefficient `-0.000749` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNDERPASS`: coefficient `0.002914` (raises CT win probability)
- `lag_03__T_bomb_zone_count`: coefficient `-0.002486` (lowers CT win probability)
- `lag_10__T_place_STAIRS`: coefficient `0.002390` (raises CT win probability)
- `lag_08__T_place_STAIRS`: coefficient `0.002181` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001928` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001853` (lowers CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.001810` (lowers CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `0.001787` (raises CT win probability)
- `lag_10__T_place_LADDER`: coefficient `-0.001780` (lowers CT win probability)
- `lag_00__T2__duck_amount`: coefficient `-0.001703` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `95412`, seconds `36.50`, LSTM delta `+0.2707`

Top all feature movements:
- `lag_08__T_place_STAIRS`: contribution `+0.041756`
- `lag_10__T_place_LADDER`: contribution `+0.040250`
- `lag_00__T_place_STAIRS`: contribution `+0.031924`
- `lag_05__T_place_JUNGLE`: contribution `+0.019975`
- `lag_00__T_shots_fired_sum`: contribution `+0.019450`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `+0.003589`
- `lag_10__T_A_site_active_infernos`: contribution `+0.001992`

### tick `95476`, seconds `37.50`, LSTM delta `+0.2704`

Top all feature movements:
- `lag_10__T_place_STAIRS`: contribution `+0.045755`
- `lag_02__T_place_STAIRS`: contribution `+0.029485`
- `lag_12__T_place_LADDER`: contribution `+0.023486`
- `lag_07__CT_place_TRAMP`: contribution `+0.019390`
- `lag_02__CT_place_TRAMP`: contribution `+0.016283`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `+0.003589`
- `lag_14__T_A_site_active_infernos`: contribution `+0.002966`

### tick `96372`, seconds `51.50`, LSTM delta `-0.2042`

Top all feature movements:
- `lag_00__CT_place_UNDERPASS`: contribution `-0.016897`
- `lag_03__T_bomb_zone_count`: contribution `-0.014472`
- `lag_10__T_shots_fired_sum`: contribution `-0.012057`
- `lag_10__T2__shots_fired`: contribution `-0.007367`
- `lag_00__T_kills_last_3s`: contribution `-0.004997`

Top utility-only movements:
- `lag_10__T2__flash_duration`: contribution `-0.003131`

### tick `95604`, seconds `39.50`, LSTM delta `-0.1755`

Top all feature movements:
- `lag_06__T_place_STAIRS`: contribution `-0.021786`
- `lag_06__CT_place_TRAMP`: contribution `-0.018102`
- `lag_11__T_place_JUNGLE`: contribution `-0.015671`
- `lag_06__T_shots_fired_sum`: contribution `-0.013505`
- `lag_00__T_shots_fired_sum`: contribution `-0.012504`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `95860`, seconds `43.50`, LSTM delta `-0.1272`

Top all feature movements:
- `lag_14__CT_place_TRAMP`: contribution `-0.015174`
- `lag_14__T_place_STAIRS`: contribution `-0.011037`
- `lag_14__T_shots_fired_sum`: contribution `-0.009983`
- `lag_01__CT_place_UNDERPASS`: contribution `+0.007742`
- `lag_00__T2__duck_amount`: contribution `-0.006509`

Top utility-only movements:
- `lag_12__T_A_site_active_infernos`: contribution `-0.003589`
