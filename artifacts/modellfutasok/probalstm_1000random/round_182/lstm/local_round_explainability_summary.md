# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mibr-bo3-vjmAHfXA4PQfROTmirSCCF/vitality-vs-mibr-m2-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `60159`, seconds `26.50`, LSTM `0.1863`, delta `-0.3297`
- tick `59615`, seconds `18.00`, LSTM `0.1670`, delta `-0.3241`
- tick `60127`, seconds `26.00`, LSTM `0.5161`, delta `+0.1792`
- tick `65023`, seconds `102.50`, LSTM `0.1379`, delta `+0.0979`
- tick `64767`, seconds `98.50`, LSTM `0.0487`, delta `-0.0814`
- tick `59487`, seconds `16.00`, LSTM `0.4401`, delta `-0.0814`
- tick `59935`, seconds `23.00`, LSTM `0.2869`, delta `+0.0729`
- tick `59807`, seconds `21.00`, LSTM `0.1716`, delta `+0.0683`
- tick `64447`, seconds `93.50`, LSTM `0.1870`, delta `+0.0541`
- tick `60223`, seconds `27.50`, LSTM `0.1151`, delta `-0.0507`

## Top 15 local ridge features

- `lag_11__T_utility_damage_last_5s`: coefficient `0.002751`, |coef| `0.002751`
- `lag_00__CT_place_QUAD`: coefficient `0.002491`, |coef| `0.002491`
- `lag_00__kill_diff_last_3s`: coefficient `0.002405`, |coef| `0.002405`
- `lag_00__CT1__is_walking`: coefficient `0.002264`, |coef| `0.002264`
- `lag_00__T_kills_last_3s`: coefficient `-0.002227`, |coef| `0.002227`
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.002205`, |coef| `0.002205`
- `lag_02__CT_place_QUAD`: coefficient `-0.002105`, |coef| `0.002105`
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001932`, |coef| `0.001932`
- `lag_00__CT_place_BANANA`: coefficient `0.001865`, |coef| `0.001865`
- `lag_10__T4__duck_amount`: coefficient `0.001802`, |coef| `0.001802`
- `lag_13__T2__duck_amount`: coefficient `0.001681`, |coef| `0.001681`
- `lag_09__CT1__is_walking`: coefficient `-0.001647`, |coef| `0.001647`
- `lag_14__CT_utility_damage_last_5s`: coefficient `-0.001505`, |coef| `0.001505`
- `lag_00__T1__is_scoped`: coefficient `0.001502`, |coef| `0.001502`
- `lag_09__CT_place_QUAD`: coefficient `-0.001448`, |coef| `0.001448`

## Top 10 utility ridge features

- `lag_11__T_utility_damage_last_5s`: coefficient `0.002751` (raises CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `0.002205` (raises CT win probability)
- `lag_04__T_utility_damage_last_5s`: coefficient `-0.001932` (lowers CT win probability)
- `lag_14__CT_utility_damage_last_5s`: coefficient `-0.001505` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.001408` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.001285` (lowers CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `0.001199` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.001096` (raises CT win probability)
- `lag_00__CT3__molly`: coefficient `0.001071` (raises CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `-0.001027` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_QUAD`: coefficient `0.002491` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002405` (raises CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.002264` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002227` (lowers CT win probability)
- `lag_02__CT_place_QUAD`: coefficient `-0.002105` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.001865` (raises CT win probability)
- `lag_10__T4__duck_amount`: coefficient `0.001802` (raises CT win probability)
- `lag_13__T2__duck_amount`: coefficient `0.001681` (raises CT win probability)
- `lag_09__CT1__is_walking`: coefficient `-0.001647` (lowers CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.001502` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `60159`, seconds `26.50`, LSTM delta `-0.3297`

Top all feature movements:
- `lag_11__T_utility_damage_last_5s`: contribution `-0.023957`
- `lag_02__CT_place_QUAD`: contribution `-0.016589`
- `lag_00__T1__is_scoped`: contribution `-0.008579`
- `lag_11__utility_damage_diff_last_5s`: contribution `-0.007075`
- `lag_00__T_kills_last_3s`: contribution `-0.007056`

Top utility-only movements:
- `lag_11__T_utility_damage_last_5s`: contribution `-0.023957`
- `lag_11__utility_damage_diff_last_5s`: contribution `-0.007075`

### tick `59615`, seconds `18.00`, LSTM delta `-0.3241`

Top all feature movements:
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.022100`
- `lag_00__CT_place_QUAD`: contribution `-0.019630`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.016827`
- `lag_09__CT_place_QUAD`: contribution `-0.011416`
- `lag_03__CT5__flash_duration`: contribution `-0.010503`

Top utility-only movements:
- `lag_04__utility_damage_diff_last_5s`: contribution `-0.022100`
- `lag_04__T_utility_damage_last_5s`: contribution `-0.016827`
- `lag_03__CT5__flash_duration`: contribution `-0.010503`
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.008285`
- `lag_03__CT_flash_duration_sum`: contribution `-0.007586`

### tick `60127`, seconds `26.00`, LSTM delta `+0.1792`

Top all feature movements:
- `lag_01__CT_place_QUAD`: contribution `+0.009346`
- `lag_00__T1__is_scoped`: contribution `+0.008579`
- `lag_10__T_utility_damage_last_5s`: contribution `+0.007055`
- `lag_10__T4__duck_amount`: contribution `+0.006663`
- `lag_13__T2__duck_amount`: contribution `+0.006428`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `+0.007055`

### tick `65023`, seconds `102.50`, LSTM delta `+0.0979`

Top all feature movements:
- `lag_02__CT_shots_fired_sum`: contribution `+0.010264`
- `lag_00__T1__is_scoped`: contribution `-0.008579`
- `lag_00__kill_diff_last_3s`: contribution `+0.005789`
- `lag_02__T3__flash_duration`: contribution `+0.004582`
- `lag_09__CT1__is_walking`: contribution `+0.003844`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.004582`
- `lag_03__T5__flash_duration`: contribution `+0.003563`
- `lag_14__T3__flash_duration`: contribution `+0.003424`
- `lag_14__T_flash_duration_sum`: contribution `+0.002681`
- `lag_14__T5__flash_duration`: contribution `+0.002574`

### tick `64767`, seconds `98.50`, LSTM delta `-0.0814`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.007056`
- `lag_00__kill_diff_last_3s`: contribution `-0.005789`
- `lag_00__CT1__is_walking`: contribution `+0.005285`
- `lag_13__CT5__duck_amount`: contribution `-0.004556`
- `lag_10__CT_flashed_players`: contribution `-0.003838`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.002602`
- `lag_10__CT2__flash_duration`: contribution `-0.002439`
- `lag_14__T_B_site_active_infernos`: contribution `-0.002273`
- `lag_06__T5__flash_duration`: contribution `-0.002123`
- `lag_15__T_B_site_active_infernos`: contribution `-0.002051`
