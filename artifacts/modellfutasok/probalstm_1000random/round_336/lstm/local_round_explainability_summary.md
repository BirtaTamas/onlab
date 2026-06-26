# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `3`

## Largest probability jumps

- tick `20317`, seconds `19.00`, LSTM `0.2656`, delta `+0.2025`
- tick `20413`, seconds `20.50`, LSTM `0.1811`, delta `-0.1788`
- tick `19133`, seconds `0.50`, LSTM `0.0364`, delta `-0.0615`
- tick `20061`, seconds `15.00`, LSTM `0.1344`, delta `-0.0608`
- tick `20349`, seconds `19.50`, LSTM `0.3178`, delta `+0.0522`
- tick `20861`, seconds `27.50`, LSTM `0.0907`, delta `-0.0517`
- tick `20509`, seconds `22.00`, LSTM `0.1310`, delta `-0.0512`
- tick `21117`, seconds `31.50`, LSTM `0.0413`, delta `-0.0505`
- tick `20381`, seconds `20.00`, LSTM `0.3599`, delta `+0.0421`
- tick `20125`, seconds `16.00`, LSTM `0.0779`, delta `-0.0411`

## Top 15 local ridge features

- `lag_05__CT_place_ELECTRICALBOX`: coefficient `0.001921`, |coef| `0.001921`
- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.001721`, |coef| `0.001721`
- `lag_13__CT_place_BACKOFB`: coefficient `0.001364`, |coef| `0.001364`
- `lag_04__CT_place_ELECTRICALBOX`: coefficient `0.001343`, |coef| `0.001343`
- `lag_06__CT_place_BACKOFB`: coefficient `-0.001328`, |coef| `0.001328`
- `lag_07__T_place_DUMPSTER`: coefficient `-0.001251`, |coef| `0.001251`
- `lag_08__CT_place_ELECTRICALBOX`: coefficient `-0.001184`, |coef| `0.001184`
- `lag_12__CT_place_BACKOFB`: coefficient `0.001077`, |coef| `0.001077`
- `lag_10__T_place_DUMPSTER`: coefficient `0.001025`, |coef| `0.001025`
- `lag_05__CT2__duck_amount`: coefficient `0.001007`, |coef| `0.001007`
- `lag_09__CT_flashed_players`: coefficient `0.000982`, |coef| `0.000982`
- `lag_09__CT_place_BACKOFB`: coefficient `0.000875`, |coef| `0.000875`
- `lag_00__kill_diff_last_3s`: coefficient `0.000846`, |coef| `0.000846`
- `lag_02__CT_place_TSIDEUPPER`: coefficient `-0.000833`, |coef| `0.000833`
- `lag_03__CT2__duck_amount`: coefficient `-0.000820`, |coef| `0.000820`

## Top 10 utility ridge features

- `lag_12__CT2__flash_duration`: coefficient `-0.000622` (lowers CT win probability)
- `lag_09__CT2__flash_duration`: coefficient `0.000614` (raises CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `0.000504` (raises CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `0.000493` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.000491` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `0.000483` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `0.000481` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `0.000476` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `-0.000452` (lowers CT win probability)
- `lag_06__T3__flash_duration`: coefficient `0.000452` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_ELECTRICALBOX`: coefficient `0.001921` (raises CT win probability)
- `lag_01__CT_place_ELECTRICALBOX`: coefficient `-0.001721` (lowers CT win probability)
- `lag_13__CT_place_BACKOFB`: coefficient `0.001364` (raises CT win probability)
- `lag_04__CT_place_ELECTRICALBOX`: coefficient `0.001343` (raises CT win probability)
- `lag_06__CT_place_BACKOFB`: coefficient `-0.001328` (lowers CT win probability)
- `lag_07__T_place_DUMPSTER`: coefficient `-0.001251` (lowers CT win probability)
- `lag_08__CT_place_ELECTRICALBOX`: coefficient `-0.001184` (lowers CT win probability)
- `lag_12__CT_place_BACKOFB`: coefficient `0.001077` (raises CT win probability)
- `lag_10__T_place_DUMPSTER`: coefficient `0.001025` (raises CT win probability)
- `lag_05__CT2__duck_amount`: coefficient `0.001007` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `20317`, seconds `19.00`, LSTM delta `+0.2025`

Top all feature movements:
- `lag_05__CT_place_ELECTRICALBOX`: contribution `+0.022332`
- `lag_01__CT_place_ELECTRICALBOX`: contribution `+0.020009`
- `lag_07__T_place_DUMPSTER`: contribution `+0.011379`
- `lag_13__CT_place_BACKOFB`: contribution `+0.007790`
- `lag_06__CT_place_BACKOFB`: contribution `+0.007582`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `+0.002823`

### tick `20413`, seconds `20.50`, LSTM delta `-0.1788`

Top all feature movements:
- `lag_04__CT_place_ELECTRICALBOX`: contribution `-0.015610`
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.013769`
- `lag_10__T_place_DUMPSTER`: contribution `-0.009317`
- `lag_02__CT_place_TSIDEUPPER`: contribution `-0.006261`
- `lag_01__T2__is_scoped`: contribution `-0.006040`

Top utility-only movements:
- `lag_00__CT2__flash_duration`: contribution `-0.003057`
- `lag_12__CT2__flash_duration`: contribution `-0.002856`
- `lag_01__T3__flash_duration`: contribution `-0.002156`

### tick `19133`, seconds `0.50`, LSTM delta `-0.0615`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002610`
- `lag_01__T_place_TSPAWN`: contribution `-0.001907`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001905`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.001876`
- `lag_01__centroid_distance_xy`: contribution `-0.001645`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.001344`
- `lag_01__smoke_inv_diff`: contribution `-0.000991`
- `lag_01__molly_inv_diff`: contribution `-0.000862`
- `lag_01__T5__utility_total`: contribution `-0.000851`
- `lag_01__flash_inv_diff`: contribution `-0.000825`

### tick `20061`, seconds `15.00`, LSTM delta `-0.0608`

Top all feature movements:
- `lag_04__CT_place_BACKOFB`: contribution `-0.004516`
- `lag_05__CT_place_BACKOFB`: contribution `-0.003277`
- `lag_13__bomb_events_last_5s`: contribution `-0.002688`
- `lag_01__CT2__flash_duration`: contribution `-0.002078`
- `lag_08__T5__duck_amount`: contribution `-0.001723`

Top utility-only movements:
- `lag_01__CT2__flash_duration`: contribution `-0.002078`
- `lag_09__T_A_site_active_infernos`: contribution `-0.000986`
- `lag_00__T3__flash_duration`: contribution `-0.000903`
- `lag_01__CT_flash_duration_sum`: contribution `-0.000715`

### tick `20349`, seconds `19.50`, LSTM delta `+0.0522`

Top all feature movements:
- `lag_06__CT_place_ELECTRICALBOX`: contribution `+0.008497`
- `lag_13__CT_place_BACKOFB`: contribution `+0.007790`
- `lag_02__CT_place_ELECTRICALBOX`: contribution `+0.006372`
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.006103`
- `lag_08__T_place_DUMPSTER`: contribution `+0.003560`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `+0.001332`
- `lag_01__T5__flash`: contribution `+0.001023`
