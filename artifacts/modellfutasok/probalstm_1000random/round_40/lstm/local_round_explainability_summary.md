# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `18`

## Largest probability jumps

- tick `153335`, seconds `26.00`, LSTM `0.6135`, delta `+0.4114`
- tick `153239`, seconds `24.50`, LSTM `0.2948`, delta `-0.1585`
- tick `153911`, seconds `35.00`, LSTM `0.6769`, delta `+0.1350`
- tick `155031`, seconds `52.50`, LSTM `0.8889`, delta `+0.1249`
- tick `152663`, seconds `15.50`, LSTM `0.4709`, delta `-0.1086`
- tick `153303`, seconds `25.50`, LSTM `0.2021`, delta `-0.1019`
- tick `153367`, seconds `26.50`, LSTM `0.5599`, delta `-0.0536`
- tick `154231`, seconds `40.00`, LSTM `0.6219`, delta `-0.0481`
- tick `154999`, seconds `52.00`, LSTM `0.7640`, delta `+0.0451`
- tick `152695`, seconds `16.00`, LSTM `0.4310`, delta `-0.0398`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003110`, |coef| `0.003110`
- `lag_04__T_place_ELECTRICALBOX`: coefficient `0.002894`, |coef| `0.002894`
- `lag_00__kill_diff_last_3s`: coefficient `0.002877`, |coef| `0.002877`
- `lag_00__T_place_ELECTRICALBOX`: coefficient `-0.002869`, |coef| `0.002869`
- `lag_01__T_place_ELECTRICALBOX`: coefficient `-0.002423`, |coef| `0.002423`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002050`, |coef| `0.002050`
- `lag_00__T_macro_A`: coefficient `-0.002050`, |coef| `0.002050`
- `lag_00__damage_diff_last_5s`: coefficient `0.002003`, |coef| `0.002003`
- `lag_10__T_A_site_active_infernos`: coefficient `-0.001786`, |coef| `0.001786`
- `lag_10__T_B_site_active_infernos`: coefficient `-0.001694`, |coef| `0.001694`
- `lag_00__T2__alive`: coefficient `-0.001676`, |coef| `0.001676`
- `lag_00__CT_damage_last_5s`: coefficient `0.001580`, |coef| `0.001580`
- `lag_00__T2__armor`: coefficient `-0.001545`, |coef| `0.001545`
- `lag_07__CT4__is_walking`: coefficient `-0.001542`, |coef| `0.001542`
- `lag_00__T2__is_walking`: coefficient `-0.001534`, |coef| `0.001534`

## Top 10 utility ridge features

- `lag_10__T_A_site_active_infernos`: coefficient `-0.001786` (lowers CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `-0.001694` (lowers CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.001437` (lowers CT win probability)
- `lag_10__T_active_infernos`: coefficient `-0.001260` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.001250` (lowers CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.001152` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `-0.001078` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001029` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.001021` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000996` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.003110` (raises CT win probability)
- `lag_04__T_place_ELECTRICALBOX`: coefficient `0.002894` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002877` (raises CT win probability)
- `lag_00__T_place_ELECTRICALBOX`: coefficient `-0.002869` (lowers CT win probability)
- `lag_01__T_place_ELECTRICALBOX`: coefficient `-0.002423` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002050` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002050` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002003` (raises CT win probability)
- `lag_00__T2__alive`: coefficient `-0.001676` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001580` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `153335`, seconds `26.00`, LSTM delta `+0.4114`

Top all feature movements:
- `lag_04__T_place_ELECTRICALBOX`: contribution `+0.075970`
- `lag_00__T_place_ELECTRICALBOX`: contribution `+0.075311`
- `lag_00__CT_kills_last_3s`: contribution `+0.017960`
- `lag_00__kill_diff_last_3s`: contribution `+0.013851`
- `lag_02__T_place_DUMPSTER`: contribution `+0.011414`

Top utility-only movements:
- `lag_03__CT4__flash_duration`: contribution `+0.003154`

### tick `153239`, seconds `24.50`, LSTM delta `-0.1585`

Top all feature movements:
- `lag_01__T_place_ELECTRICALBOX`: contribution `-0.063604`
- `lag_03__T_flashed_players`: contribution `-0.007660`
- `lag_00__T_place_LONGDOG`: contribution `-0.004755`
- `lag_06__T_place_LONGDOG`: contribution `-0.004021`
- `lag_03__T_flash_duration_sum`: contribution `-0.003700`

Top utility-only movements:
- `lag_03__T_flash_duration_sum`: contribution `-0.003700`
- `lag_00__CT4__flash_duration`: contribution `-0.003197`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.002439`
- `lag_03__T5__flash_duration`: contribution `-0.002280`
- `lag_03__T3__flash_duration`: contribution `-0.002228`

### tick `153911`, seconds `35.00`, LSTM delta `+0.1350`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008980`
- `lag_14__CT_place_LONGDOG`: contribution `+0.007257`
- `lag_00__kill_diff_last_3s`: contribution `+0.006926`
- `lag_05__CT_place_BACKOFB`: contribution `+0.004899`
- `lag_12__CT_kills_last_3s`: contribution `+0.004841`

Top utility-only movements:
- `lag_12__CT4__flash_duration`: contribution `+0.003172`
- `lag_05__T3__flash_duration`: contribution `+0.002896`

### tick `155031`, seconds `52.50`, LSTM delta `+0.1249`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.008980`
- `lag_00__kill_diff_last_3s`: contribution `+0.006926`
- `lag_11__T_A_site_active_infernos`: contribution `+0.004277`
- `lag_00__T2__alive`: contribution `+0.004018`
- `lag_06__CT4__duck_amount`: contribution `+0.003818`

Top utility-only movements:
- `lag_11__T_A_site_active_infernos`: contribution `+0.004277`
- `lag_11__T_B_site_active_infernos`: contribution `+0.003535`
- `lag_11__T_active_infernos`: contribution `+0.002127`

### tick `152663`, seconds `15.50`, LSTM delta `-0.1086`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `-0.007733`
- `lag_00__kill_diff_last_3s`: contribution `-0.006926`
- `lag_14__T_place_DUMPSTER`: contribution `-0.006245`
- `lag_03__T_place_LONGDOG`: contribution `-0.006130`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.005246`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.003428`
