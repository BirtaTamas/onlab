# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `9`

## Largest probability jumps

- tick `74980`, seconds `86.50`, LSTM `0.5849`, delta `-0.3128`
- tick `73860`, seconds `69.00`, LSTM `0.6001`, delta `+0.2445`
- tick `74020`, seconds `71.50`, LSTM `0.8804`, delta `+0.1812`
- tick `72420`, seconds `46.50`, LSTM `0.1531`, delta `-0.1213`
- tick `72452`, seconds `47.00`, LSTM `0.2334`, delta `+0.0803`
- tick `69476`, seconds `0.50`, LSTM `0.1468`, delta `-0.0655`
- tick `73796`, seconds `68.00`, LSTM `0.3210`, delta `-0.0608`
- tick `73700`, seconds `66.50`, LSTM `0.4310`, delta `-0.0586`
- tick `69508`, seconds `1.00`, LSTM `0.2050`, delta `+0.0582`
- tick `73348`, seconds `61.00`, LSTM `0.3932`, delta `-0.0555`

## Top 15 local ridge features

- `lag_00__CT_place_MAIN`: coefficient `0.005897`, |coef| `0.005897`
- `lag_00__kill_diff_last_3s`: coefficient `0.003352`, |coef| `0.003352`
- `lag_00__T_kills_last_3s`: coefficient `-0.003232`, |coef| `0.003232`
- `lag_00__damage_diff_last_5s`: coefficient `0.002882`, |coef| `0.002882`
- `lag_00__T_damage_last_5s`: coefficient `-0.002184`, |coef| `0.002184`
- `lag_09__T_place_MAIN`: coefficient `-0.001972`, |coef| `0.001972`
- `lag_10__T_place_MAIN`: coefficient `-0.001471`, |coef| `0.001471`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001452`, |coef| `0.001452`
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.001433`, |coef| `0.001433`
- `lag_03__CT_place_MAIN`: coefficient `0.001381`, |coef| `0.001381`
- `lag_07__T_duck_amount_mean`: coefficient `-0.001376`, |coef| `0.001376`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001355`, |coef| `0.001355`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001331`, |coef| `0.001331`
- `lag_02__T_place_MAIN`: coefficient `-0.001330`, |coef| `0.001330`
- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001313`, |coef| `0.001313`

## Top 10 utility ridge features

- `lag_09__CT_A_site_active_infernos`: coefficient `-0.001433` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001205` (raises CT win probability)
- `lag_11__CT2__molly`: coefficient `0.001110` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.001087` (raises CT win probability)
- `lag_05__CT4__flash`: coefficient `-0.001041` (lowers CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001029` (lowers CT win probability)
- `lag_09__CT_active_infernos`: coefficient `-0.000949` (lowers CT win probability)
- `lag_01__T_flash_duration_sum`: coefficient `0.000938` (raises CT win probability)
- `lag_08__T_A_site_active_infernos`: coefficient `0.000931` (raises CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.000817` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_MAIN`: coefficient `0.005897` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003352` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003232` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002882` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002184` (lowers CT win probability)
- `lag_09__T_place_MAIN`: coefficient `-0.001972` (lowers CT win probability)
- `lag_10__T_place_MAIN`: coefficient `-0.001471` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001452` (lowers CT win probability)
- `lag_03__CT_place_MAIN`: coefficient `0.001381` (raises CT win probability)
- `lag_07__T_duck_amount_mean`: coefficient `-0.001376` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `74980`, seconds `86.50`, LSTM delta `-0.3128`

Top all feature movements:
- `lag_00__CT_place_MAIN`: contribution `-0.079412`
- `lag_00__T_kills_last_3s`: contribution `-0.020479`
- `lag_00__kill_diff_last_3s`: contribution `-0.016137`
- `lag_00__damage_diff_last_5s`: contribution `-0.010859`
- `lag_00__T_damage_last_5s`: contribution `-0.008745`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.005058`
- `lag_11__CT2__molly`: contribution `-0.002737`

### tick `73860`, seconds `69.00`, LSTM delta `+0.2445`

Top all feature movements:
- `lag_09__T_place_MAIN`: contribution `+0.012747`
- `lag_10__T_place_MAIN`: contribution `+0.009507`
- `lag_02__T_place_MAIN`: contribution `+0.008596`
- `lag_00__kill_diff_last_3s`: contribution `+0.008069`
- `lag_01__T4__flash_duration`: contribution `+0.006806`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.006806`
- `lag_00__T4__flash_duration`: contribution `+0.005814`
- `lag_01__T3__flash_duration`: contribution `+0.004272`
- `lag_01__T_flash_duration_sum`: contribution `+0.003690`
- `lag_05__CT4__flash`: contribution `+0.003609`

### tick `74020`, seconds `71.50`, LSTM delta `+0.1812`

Top all feature movements:
- `lag_03__CT_place_MAIN`: contribution `+0.009302`
- `lag_00__kill_diff_last_3s`: contribution `+0.008069`
- `lag_00__damage_diff_last_5s`: contribution `+0.006763`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006474`
- `lag_04__CT_place_MAIN`: contribution `+0.006037`

Top utility-only movements:
- `lag_06__T4__flash_duration`: contribution `+0.004616`
- `lag_05__T4__flash_duration`: contribution `+0.003281`

### tick `72420`, seconds `46.50`, LSTM delta `-0.1213`

Top all feature movements:
- `lag_00__CT_place_MAIN`: contribution `-0.039706`
- `lag_00__T_kills_last_3s`: contribution `-0.010240`
- `lag_00__T2__is_scoped`: contribution `-0.009409`
- `lag_00__kill_diff_last_3s`: contribution `-0.008069`
- `lag_00__T_damage_last_5s`: contribution `-0.005237`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `72452`, seconds `47.00`, LSTM delta `+0.0803`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.008069`
- `lag_01__T2__is_scoped`: contribution `+0.007812`
- `lag_01__CT_place_MAIN`: contribution `-0.006678`
- `lag_00__damage_diff_last_5s`: contribution `+0.004877`
- `lag_00__CT_kills_last_3s`: contribution `+0.003105`

Top utility-only movements:
- No utility movement among the top local contributors.
