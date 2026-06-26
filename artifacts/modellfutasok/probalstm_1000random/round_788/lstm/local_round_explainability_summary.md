# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `3`

## Largest probability jumps

- tick `23022`, seconds `63.50`, LSTM `0.0280`, delta `-0.0835`
- tick `18990`, seconds `0.50`, LSTM `0.0892`, delta `-0.0647`
- tick `22926`, seconds `62.00`, LSTM `0.1001`, delta `-0.0308`
- tick `19374`, seconds `6.50`, LSTM `0.1650`, delta `+0.0237`
- tick `19406`, seconds `7.00`, LSTM `0.1463`, delta `-0.0187`
- tick `19566`, seconds `9.50`, LSTM `0.1453`, delta `-0.0173`
- tick `20750`, seconds `28.00`, LSTM `0.1399`, delta `+0.0168`
- tick `19214`, seconds `4.00`, LSTM `0.1102`, delta `+0.0167`
- tick `22190`, seconds `50.50`, LSTM `0.1350`, delta `-0.0155`
- tick `19182`, seconds `3.50`, LSTM `0.0935`, delta `+0.0153`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001116`, |coef| `0.001116`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000955`, |coef| `0.000955`
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000618`, |coef| `0.000618`
- `lag_00__T_place_CANAL`: coefficient `0.000603`, |coef| `0.000603`
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000526`, |coef| `0.000526`
- `lag_13__T5__duck_amount`: coefficient `-0.000522`, |coef| `0.000522`
- `lag_00__CT_place_BACKOFB`: coefficient `0.000508`, |coef| `0.000508`
- `lag_09__T5__duck_amount`: coefficient `0.000445`, |coef| `0.000445`
- `lag_02__T_place_MAIN`: coefficient `-0.000430`, |coef| `0.000430`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000425`, |coef| `0.000425`
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000421`, |coef| `0.000421`
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.000410`, |coef| `0.000410`
- `lag_06__CT_place_MAIN`: coefficient `-0.000408`, |coef| `0.000408`
- `lag_00__T_place_MAIN`: coefficient `0.000407`, |coef| `0.000407`
- `lag_03__T_active_infernos`: coefficient `-0.000405`, |coef| `0.000405`

## Top 10 utility ridge features

- `lag_01__CT_flash_alpha_mean`: coefficient `0.000618` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000526` (lowers CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `-0.000421` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.000405` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `0.000330` (raises CT win probability)
- `lag_06__T_active_infernos`: coefficient `-0.000322` (lowers CT win probability)
- `lag_07__T3__molly`: coefficient `0.000321` (raises CT win probability)
- `lag_07__CT5__smoke`: coefficient `0.000317` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.000307` (lowers CT win probability)
- `lag_09__T3__smoke`: coefficient `0.000298` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001116` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000955` (raises CT win probability)
- `lag_00__T_place_CANAL`: coefficient `0.000603` (raises CT win probability)
- `lag_13__T5__duck_amount`: coefficient `-0.000522` (lowers CT win probability)
- `lag_00__CT_place_BACKOFB`: coefficient `0.000508` (raises CT win probability)
- `lag_09__T5__duck_amount`: coefficient `0.000445` (raises CT win probability)
- `lag_02__T_place_MAIN`: coefficient `-0.000430` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000425` (lowers CT win probability)
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.000410` (lowers CT win probability)
- `lag_06__CT_place_MAIN`: coefficient `-0.000408` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `23022`, seconds `63.50`, LSTM delta `-0.0835`

Top all feature movements:
- `lag_00__T_place_MAIN`: contribution `-0.005261`
- `lag_00__CT_place_CTSIDEUPPER`: contribution `-0.004931`
- `lag_02__T_place_MAIN`: contribution `-0.002781`
- `lag_00__T_shots_fired_sum`: contribution `-0.002549`
- `lag_13__T_place_MAIN`: contribution `-0.002152`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.001565`
- `lag_06__T_A_site_active_infernos`: contribution `-0.001253`
- `lag_02__T_A_site_active_infernos`: contribution `-0.000913`
- `lag_03__T_active_infernos`: contribution `-0.000844`

### tick `18990`, seconds `0.50`, LSTM delta `-0.0647`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.028751`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.004600`
- `lag_01__T_place_TSPAWN`: contribution `-0.001539`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001135`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000914`

Top utility-only movements:
- `lag_01__CT_flash_alpha_mean`: contribution `-0.004600`
- `lag_01__molly_inv_diff`: contribution `-0.000606`
- `lag_01__T_molly_inv`: contribution `-0.000393`
- `lag_01__utility_inv_diff`: contribution `-0.000319`
- `lag_01__T_utility_inv`: contribution `-0.000301`

### tick `22926`, seconds `62.00`, LSTM delta `-0.0308`

Top all feature movements:
- `lag_09__T5__duck_amount`: contribution `-0.001691`
- `lag_03__T_A_site_active_infernos`: contribution `-0.001565`
- `lag_14__T_place_MAIN`: contribution `-0.001402`
- `lag_15__CT_place_MAIN`: contribution `-0.001354`
- `lag_00__T_shots_fired_sum`: contribution `-0.001275`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.001565`
- `lag_03__T_active_infernos`: contribution `-0.000844`
- `lag_00__T2__flash`: contribution `-0.000486`

### tick `19374`, seconds `6.50`, LSTM delta `+0.0237`

Top all feature movements:
- `lag_13__CT_place_CTSIDEUPPER`: contribution `+0.007676`
- `lag_10__T_he_last_5s`: contribution `+0.003530`
- `lag_09__CT_place_CTSIDEUPPER`: contribution `+0.002201`
- `lag_10__CT_place_CTSIDEUPPER`: contribution `-0.002150`
- `lag_00__T_he_last_5s`: contribution `+0.001956`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `+0.003530`
- `lag_00__T_he_last_5s`: contribution `+0.001956`

### tick `19406`, seconds `7.00`, LSTM delta `-0.0187`

Top all feature movements:
- `lag_11__CT_place_CTSIDEUPPER`: contribution `-0.003204`
- `lag_14__CT_place_CTSIDEUPPER`: contribution `-0.003165`
- `lag_01__T_he_last_5s`: contribution `-0.002527`
- `lag_10__CT_place_CTSIDEUPPER`: contribution `-0.002150`
- `lag_07__CT_place_PALACEINTERIOR`: contribution `-0.001570`

Top utility-only movements:
- `lag_01__T_he_last_5s`: contribution `-0.002527`
- `lag_14__CT_flash_alpha_mean`: contribution `-0.001102`
