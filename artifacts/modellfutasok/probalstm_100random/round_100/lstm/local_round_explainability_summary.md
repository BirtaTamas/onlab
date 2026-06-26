# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `11`

## Largest probability jumps

- tick `91975`, seconds `77.00`, LSTM `0.6666`, delta `+0.2064`
- tick `92103`, seconds `79.00`, LSTM `0.9013`, delta `+0.2056`
- tick `91623`, seconds `71.50`, LSTM `0.4913`, delta `-0.1363`
- tick `91751`, seconds `73.50`, LSTM `0.3657`, delta `-0.0958`
- tick `91559`, seconds `70.50`, LSTM `0.6298`, delta `+0.0948`
- tick `91847`, seconds `75.00`, LSTM `0.4263`, delta `+0.0619`
- tick `91943`, seconds `76.50`, LSTM `0.4602`, delta `+0.0526`
- tick `91783`, seconds `74.00`, LSTM `0.3132`, delta `-0.0525`
- tick `91815`, seconds `74.50`, LSTM `0.3645`, delta `+0.0513`
- tick `92071`, seconds `78.50`, LSTM `0.6957`, delta `+0.0511`

## Top 15 local ridge features

- `lag_09__T_place_STAIRS`: coefficient `0.002308`, |coef| `0.002308`
- `lag_05__T_place_STAIRS`: coefficient `0.001725`, |coef| `0.001725`
- `lag_11__CT_place_STAIRS`: coefficient `-0.001659`, |coef| `0.001659`
- `lag_00__kill_diff_last_3s`: coefficient `0.001401`, |coef| `0.001401`
- `lag_08__T_place_STAIRS`: coefficient `0.001322`, |coef| `0.001322`
- `lag_02__T_place_STAIRS`: coefficient `-0.001190`, |coef| `0.001190`
- `lag_00__CT_kills_last_3s`: coefficient `0.001184`, |coef| `0.001184`
- `lag_10__T_shots_fired_sum`: coefficient `0.001161`, |coef| `0.001161`
- `lag_00__damage_diff_last_5s`: coefficient `0.001140`, |coef| `0.001140`
- `lag_15__CT_place_STAIRS`: coefficient `-0.001137`, |coef| `0.001137`
- `lag_01__T3__is_scoped`: coefficient `0.001068`, |coef| `0.001068`
- `lag_06__T_place_STAIRS`: coefficient `0.001015`, |coef| `0.001015`
- `lag_11__T_place_STAIRS`: coefficient `0.000996`, |coef| `0.000996`
- `lag_12__T_place_STAIRS`: coefficient `0.000988`, |coef| `0.000988`
- `lag_10__T5__flash_duration`: coefficient `-0.000956`, |coef| `0.000956`

## Top 10 utility ridge features

- `lag_10__T5__flash_duration`: coefficient `-0.000956` (lowers CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.000896` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `-0.000854` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000823` (lowers CT win probability)
- `lag_07__T3__flash_duration`: coefficient `-0.000817` (lowers CT win probability)
- `lag_03__T3__flash_duration`: coefficient `-0.000786` (lowers CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.000766` (lowers CT win probability)
- `lag_06__T3__flash_duration`: coefficient `-0.000752` (lowers CT win probability)
- `lag_14__CT_flash_duration_sum`: coefficient `-0.000741` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.000734` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_STAIRS`: coefficient `0.002308` (raises CT win probability)
- `lag_05__T_place_STAIRS`: coefficient `0.001725` (raises CT win probability)
- `lag_11__CT_place_STAIRS`: coefficient `-0.001659` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001401` (raises CT win probability)
- `lag_08__T_place_STAIRS`: coefficient `0.001322` (raises CT win probability)
- `lag_02__T_place_STAIRS`: coefficient `-0.001190` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001184` (raises CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `0.001161` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001140` (raises CT win probability)
- `lag_15__CT_place_STAIRS`: coefficient `-0.001137` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `91975`, seconds `77.00`, LSTM delta `+0.2064`

Top all feature movements:
- `lag_05__T_place_STAIRS`: contribution `+0.033016`
- `lag_11__CT_place_STAIRS`: contribution `+0.012910`
- `lag_02__T4__shots_fired`: contribution `+0.011839`
- `lag_02__T_shots_fired_sum`: contribution `+0.010342`
- `lag_01__T3__is_scoped`: contribution `+0.006849`

Top utility-only movements:
- `lag_14__T3__flash_duration`: contribution `+0.005978`
- `lag_02__T3__flash_duration`: contribution `+0.005109`
- `lag_14__T5__flash_duration`: contribution `+0.003718`
- `lag_10__CT3__flash_duration`: contribution `+0.002984`

### tick `92103`, seconds `79.00`, LSTM delta `+0.2056`

Top all feature movements:
- `lag_09__T_place_STAIRS`: contribution `+0.044191`
- `lag_02__T_place_STAIRS`: contribution `+0.022781`
- `lag_15__CT_place_STAIRS`: contribution `+0.008849`
- `lag_06__T4__shots_fired`: contribution `+0.008420`
- `lag_06__T_shots_fired_sum`: contribution `+0.007067`

Top utility-only movements:
- `lag_06__T3__flash_duration`: contribution `+0.005018`
- `lag_14__CT3__flash_duration`: contribution `+0.003779`
- `lag_14__CT_flash_duration_sum`: contribution `+0.001524`

### tick `91623`, seconds `71.50`, LSTM delta `-0.1363`

Top all feature movements:
- `lag_11__CT_place_STAIRS`: contribution `-0.012910`
- `lag_00__CT_place_STAIRS`: contribution `-0.007154`
- `lag_10__T5__flash_duration`: contribution `-0.005643`
- `lag_03__T3__flash_duration`: contribution `-0.005245`
- `lag_00__kill_diff_last_3s`: contribution `-0.003373`

Top utility-only movements:
- `lag_10__T5__flash_duration`: contribution `-0.005643`
- `lag_03__T3__flash_duration`: contribution `-0.005245`
- `lag_12__CT5__flash_duration`: contribution `-0.003311`
- `lag_03__T5__flash_duration`: contribution `-0.002372`
- `lag_10__CT3__flash_duration`: contribution `-0.002329`

### tick `91751`, seconds `73.50`, LSTM delta `-0.0958`

Top all feature movements:
- `lag_15__CT_place_STAIRS`: contribution `-0.008849`
- `lag_07__T3__flash_duration`: contribution `-0.005451`
- `lag_14__T5__flash_duration`: contribution `-0.005038`
- `lag_04__CT_place_STAIRS`: contribution `-0.004065`
- `lag_00__CT_kills_last_3s`: contribution `-0.003420`

Top utility-only movements:
- `lag_07__T3__flash_duration`: contribution `-0.005451`
- `lag_14__T5__flash_duration`: contribution `-0.005038`
- `lag_14__CT3__flash_duration`: contribution `-0.002949`
- `lag_14__CT_flash_duration_sum`: contribution `-0.002136`
- `lag_07__T5__flash_duration`: contribution `-0.001802`

### tick `91559`, seconds `70.50`, LSTM delta `+0.0948`

Top all feature movements:
- `lag_09__CT_place_STAIRS`: contribution `+0.003841`
- `lag_08__T5__flash_duration`: contribution `+0.003737`
- `lag_00__CT_kills_last_3s`: contribution `+0.003420`
- `lag_00__kill_diff_last_3s`: contribution `+0.003373`
- `lag_03__CT5__duck_amount`: contribution `+0.003219`

Top utility-only movements:
- `lag_08__T5__flash_duration`: contribution `+0.003737`
- `lag_10__CT3__flash_duration`: contribution `-0.001814`
- `lag_01__T5__flash_duration`: contribution `+0.001575`
- `lag_00__T_A_site_active_infernos`: contribution `+0.001472`
- `lag_06__T_A_site_active_infernos`: contribution `+0.001263`
