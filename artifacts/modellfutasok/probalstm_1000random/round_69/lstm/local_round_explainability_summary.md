# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-872ZDvS9tk2PrtGeXVe8dJ/aurora-vs-heroic-m1-train-p3.csv`
- round_num: `2`

## Largest probability jumps

- tick `16745`, seconds `97.50`, LSTM `0.1035`, delta `-0.2777`
- tick `15881`, seconds `84.00`, LSTM `0.7335`, delta `+0.2149`
- tick `16265`, seconds `90.00`, LSTM `0.6763`, delta `+0.1810`
- tick `16425`, seconds `92.50`, LSTM `0.5060`, delta `-0.1751`
- tick `16137`, seconds `88.00`, LSTM `0.5248`, delta `-0.1739`
- tick `16393`, seconds `92.00`, LSTM `0.6811`, delta `+0.0458`
- tick `16681`, seconds `96.50`, LSTM `0.4011`, delta `-0.0433`
- tick `15945`, seconds `85.00`, LSTM `0.7394`, delta `+0.0277`
- tick `16361`, seconds `91.50`, LSTM `0.6353`, delta `-0.0277`
- tick `16809`, seconds `98.50`, LSTM `0.0599`, delta `-0.0276`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002782`, |coef| `0.002782`
- `lag_00__damage_diff_last_5s`: coefficient `0.002455`, |coef| `0.002455`
- `lag_02__T1__duck_amount`: coefficient `-0.002004`, |coef| `0.002004`
- `lag_00__CT_kills_last_3s`: coefficient `0.001908`, |coef| `0.001908`
- `lag_03__CT1__flash_duration`: coefficient `0.001856`, |coef| `0.001856`
- `lag_02__T2__is_scoped`: coefficient `0.001730`, |coef| `0.001730`
- `lag_02__CT_place_BACKOFB`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_09__T_place_BACKOFB`: coefficient `0.001595`, |coef| `0.001595`
- `lag_00__T_kills_last_3s`: coefficient `-0.001568`, |coef| `0.001568`
- `lag_03__T1__duck_amount`: coefficient `-0.001489`, |coef| `0.001489`
- `lag_01__T1__duck_amount`: coefficient `-0.001486`, |coef| `0.001486`
- `lag_03__CT_flashed_players`: coefficient `0.001421`, |coef| `0.001421`
- `lag_02__T_B_site_active_infernos`: coefficient `-0.001411`, |coef| `0.001411`
- `lag_03__T_bomb_zone_count`: coefficient `-0.001375`, |coef| `0.001375`
- `lag_00__T_damage_last_5s`: coefficient `-0.001363`, |coef| `0.001363`

## Top 10 utility ridge features

- `lag_03__CT1__flash_duration`: coefficient `0.001856` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.001411` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.001337` (raises CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.001292` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `-0.001267` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.001045` (raises CT win probability)
- `lag_06__CT_B_site_active_infernos`: coefficient `0.001035` (raises CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.001034` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000923` (lowers CT win probability)
- `lag_10__CT1__molly`: coefficient `-0.000866` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002782` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002455` (raises CT win probability)
- `lag_02__T1__duck_amount`: coefficient `-0.002004` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001908` (raises CT win probability)
- `lag_02__T2__is_scoped`: coefficient `0.001730` (raises CT win probability)
- `lag_02__CT_place_BACKOFB`: coefficient `-0.001712` (lowers CT win probability)
- `lag_09__T_place_BACKOFB`: coefficient `0.001595` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001568` (lowers CT win probability)
- `lag_03__T1__duck_amount`: coefficient `-0.001489` (lowers CT win probability)
- `lag_01__T1__duck_amount`: coefficient `-0.001486` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `16745`, seconds `97.50`, LSTM delta `-0.2777`

Top all feature movements:
- `lag_12__T2__is_scoped`: contribution `-0.011880`
- `lag_11__CT_place_ENTRANCE`: contribution `-0.010914`
- `lag_02__CT_place_BACKOFB`: contribution `-0.009775`
- `lag_03__T_bomb_zone_count`: contribution `-0.008002`
- `lag_12__CT_place_LONGDOG`: contribution `-0.007854`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `-0.003991`
- `lag_02__T_A_site_active_infernos`: contribution `-0.003771`

### tick `15881`, seconds `84.00`, LSTM delta `+0.2149`

Top all feature movements:
- `lag_02__T2__is_scoped`: contribution `+0.015247`
- `lag_03__CT1__flash_duration`: contribution `+0.008696`
- `lag_00__kill_diff_last_3s`: contribution `+0.006695`
- `lag_03__CT_flashed_players`: contribution `+0.006224`
- `lag_00__CT_kills_last_3s`: contribution `+0.005508`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `+0.008696`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.003556`
- `lag_12__CT_B_site_active_infernos`: contribution `+0.003552`
- `lag_12__CT_A_site_active_infernos`: contribution `+0.002845`

### tick `16265`, seconds `90.00`, LSTM delta `+0.1810`

Top all feature movements:
- `lag_14__T2__is_scoped`: contribution `+0.011156`
- `lag_00__kill_diff_last_3s`: contribution `+0.006695`
- `lag_11__T_place_IVY`: contribution `+0.006621`
- `lag_15__CT1__flash_duration`: contribution `+0.006055`
- `lag_00__CT_kills_last_3s`: contribution `+0.005508`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `+0.006055`
- `lag_02__T_B_site_active_infernos`: contribution `+0.003991`
- `lag_02__T_A_site_active_infernos`: contribution `+0.003771`
- `lag_02__T_active_infernos`: contribution `+0.001800`
- `lag_05__CT1__flash_duration`: contribution `+0.001730`

### tick `16425`, seconds `92.50`, LSTM delta `-0.1751`

Top all feature movements:
- `lag_02__T2__is_scoped`: contribution `-0.015247`
- `lag_01__CT_place_ENTRANCE`: contribution `-0.007681`
- `lag_00__kill_diff_last_3s`: contribution `-0.006695`
- `lag_00__damage_diff_last_5s`: contribution `-0.005539`
- `lag_02__CT_place_LONGDOG`: contribution `+0.005263`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `-0.001861`

### tick `16137`, seconds `88.00`, LSTM delta `-0.1739`

Top all feature movements:
- `lag_07__T_place_IVY`: contribution `-0.006811`
- `lag_00__kill_diff_last_3s`: contribution `-0.006695`
- `lag_01__CT1__flash_duration`: contribution `-0.005626`
- `lag_01__T1__duck_amount`: contribution `-0.005107`
- `lag_00__T_kills_last_3s`: contribution `-0.004966`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `-0.005626`
- `lag_11__CT1__flash_duration`: contribution `-0.004326`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002733`
- `lag_14__CT_B_site_active_infernos`: contribution `-0.002321`
