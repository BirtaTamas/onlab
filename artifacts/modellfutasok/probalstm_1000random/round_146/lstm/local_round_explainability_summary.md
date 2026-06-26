# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `5375`, seconds `64.50`, LSTM `0.8069`, delta `+0.2990`
- tick `5311`, seconds `63.50`, LSTM `0.4750`, delta `+0.2358`
- tick `3775`, seconds `39.50`, LSTM `0.5238`, delta `-0.1938`
- tick `3199`, seconds `30.50`, LSTM `0.8667`, delta `+0.1616`
- tick `3135`, seconds `29.50`, LSTM `0.6516`, delta `+0.1491`
- tick `3295`, seconds `32.00`, LSTM `0.8253`, delta `-0.1045`
- tick `4159`, seconds `45.50`, LSTM `0.2377`, delta `-0.0805`
- tick `5439`, seconds `65.50`, LSTM `0.8963`, delta `+0.0627`
- tick `3167`, seconds `30.00`, LSTM `0.7050`, delta `+0.0535`
- tick `3231`, seconds `31.00`, LSTM `0.9113`, delta `+0.0446`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.006280`, |coef| `0.006280`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.006087`, |coef| `0.006087`
- `lag_00__kill_diff_last_3s`: coefficient `0.005870`, |coef| `0.005870`
- `lag_00__CT_damage_last_5s`: coefficient `0.004203`, |coef| `0.004203`
- `lag_00__damage_diff_last_5s`: coefficient `0.004172`, |coef| `0.004172`
- `lag_00__T_place_TRAMP`: coefficient `-0.003681`, |coef| `0.003681`
- `lag_03__T_duck_amount_mean`: coefficient `-0.003496`, |coef| `0.003496`
- `lag_12__T_duck_amount_mean`: coefficient `-0.003297`, |coef| `0.003297`
- `lag_10__T_duck_amount_mean`: coefficient `-0.003040`, |coef| `0.003040`
- `lag_02__CT_kills_last_3s`: coefficient `0.002975`, |coef| `0.002975`
- `lag_11__CT5__duck_amount`: coefficient `0.002957`, |coef| `0.002957`
- `lag_13__CT5__duck_amount`: coefficient `0.002935`, |coef| `0.002935`
- `lag_00__T2__alive`: coefficient `-0.002912`, |coef| `0.002912`
- `lag_00__T2__hp`: coefficient `-0.002876`, |coef| `0.002876`
- `lag_02__CT_damage_last_5s`: coefficient `0.002732`, |coef| `0.002732`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.006087` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.002346` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001863` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001637` (lowers CT win probability)
- `lag_02__T2__flash`: coefficient `-0.001456` (lowers CT win probability)
- `lag_14__T_A_site_active_smokes`: coefficient `-0.001185` (lowers CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `-0.001144` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.001035` (lowers CT win probability)
- `lag_04__T_flash_alpha_mean`: coefficient `-0.001034` (lowers CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `-0.001030` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.006280` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005870` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.004203` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.004172` (raises CT win probability)
- `lag_00__T_place_TRAMP`: coefficient `-0.003681` (lowers CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.003496` (lowers CT win probability)
- `lag_12__T_duck_amount_mean`: coefficient `-0.003297` (lowers CT win probability)
- `lag_10__T_duck_amount_mean`: coefficient `-0.003040` (lowers CT win probability)
- `lag_02__CT_kills_last_3s`: coefficient `0.002975` (raises CT win probability)
- `lag_11__CT5__duck_amount`: coefficient `0.002957` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `5375`, seconds `64.50`, LSTM delta `+0.2990`

Top all feature movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.036930`
- `lag_00__CT_kills_last_3s`: contribution `+0.018130`
- `lag_00__kill_diff_last_3s`: contribution `+0.014130`
- `lag_13__CT5__duck_amount`: contribution `+0.011079`
- `lag_00__T_place_TRAMP`: contribution `+0.010774`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.036930`

### tick `5311`, seconds `63.50`, LSTM delta `+0.2358`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.018130`
- `lag_00__kill_diff_last_3s`: contribution `+0.014130`
- `lag_02__T_duck_amount_mean`: contribution `-0.013182`
- `lag_11__CT5__duck_amount`: contribution `+0.011162`
- `lag_03__T_duck_amount_mean`: contribution `+0.010166`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3775`, seconds `39.50`, LSTM delta `-0.1938`

Top all feature movements:
- `lag_13__CT_place_SCAFFOLDING`: contribution `-0.024655`
- `lag_00__kill_diff_last_3s`: contribution `-0.014130`
- `lag_00__damage_diff_last_5s`: contribution `-0.009411`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.009389`
- `lag_00__T_duck_amount_mean`: contribution `+0.007820`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `3199`, seconds `30.50`, LSTM delta `+0.1616`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.018130`
- `lag_00__kill_diff_last_3s`: contribution `+0.014130`
- `lag_00__T_place_TRAMP`: contribution `+0.010774`
- `lag_06__CT_place_STAIRS`: contribution `+0.009442`
- `lag_02__CT_kills_last_3s`: contribution `+0.008589`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `+0.002089`

### tick `3135`, seconds `29.50`, LSTM delta `+0.1491`

Top all feature movements:
- `lag_10__CT_place_SCAFFOLDING`: contribution `+0.035410`
- `lag_00__CT_kills_last_3s`: contribution `+0.018130`
- `lag_00__kill_diff_last_3s`: contribution `+0.014130`
- `lag_00__T_place_TRAMP`: contribution `+0.010774`
- `lag_14__CT_place_STAIRS`: contribution `+0.008068`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.002376`
