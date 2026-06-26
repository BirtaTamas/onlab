# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `8283`, seconds `16.50`, LSTM `0.5389`, delta `+0.3696`
- tick `9595`, seconds `37.00`, LSTM `0.2689`, delta `-0.2364`
- tick `8123`, seconds `14.00`, LSTM `0.3002`, delta `+0.2147`
- tick `7995`, seconds `12.00`, LSTM `0.3262`, delta `-0.1896`
- tick `8027`, seconds `12.50`, LSTM `0.1406`, delta `-0.1856`
- tick `8859`, seconds `25.50`, LSTM `0.3650`, delta `-0.1405`
- tick `8827`, seconds `25.00`, LSTM `0.5055`, delta `-0.1246`
- tick `8795`, seconds `24.50`, LSTM `0.6300`, delta `+0.1087`
- tick `8923`, seconds `26.50`, LSTM `0.4527`, delta `+0.1024`
- tick `8347`, seconds `17.50`, LSTM `0.4823`, delta `-0.0713`

## Top 15 local ridge features

- `lag_13__T_place_HUT`: coefficient `0.004181`, |coef| `0.004181`
- `lag_08__CT_place_RAFTERS`: coefficient `-0.003489`, |coef| `0.003489`
- `lag_00__kill_diff_last_3s`: coefficient `0.003033`, |coef| `0.003033`
- `lag_14__T_damage_last_5s`: coefficient `0.002973`, |coef| `0.002973`
- `lag_13__CT_place_HELL`: coefficient `-0.002958`, |coef| `0.002958`
- `lag_00__CT5__alive`: coefficient `0.002732`, |coef| `0.002732`
- `lag_12__T_place_HUT`: coefficient `0.002717`, |coef| `0.002717`
- `lag_02__CT_place_HUTROOF`: coefficient `-0.002659`, |coef| `0.002659`
- `lag_00__damage_diff_last_5s`: coefficient `0.002571`, |coef| `0.002571`
- `lag_00__CT_place_HEAVEN`: coefficient `0.002546`, |coef| `0.002546`
- `lag_00__T_kills_last_3s`: coefficient `-0.002471`, |coef| `0.002471`
- `lag_00__T_place_HUT`: coefficient `-0.002458`, |coef| `0.002458`
- `lag_13__T_bomb_zone_count`: coefficient `0.002448`, |coef| `0.002448`
- `lag_15__T_bomb_zone_count`: coefficient `0.002417`, |coef| `0.002417`
- `lag_11__T_place_HUT`: coefficient `0.002358`, |coef| `0.002358`

## Top 10 utility ridge features

- `lag_05__T2__flash_duration`: coefficient `-0.001864` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.001661` (raises CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.001622` (raises CT win probability)
- `lag_07__T_B_site_active_smokes`: coefficient `0.001564` (raises CT win probability)
- `lag_07__T_A_site_active_smokes`: coefficient `0.001472` (raises CT win probability)
- `lag_10__T_B_site_active_smokes`: coefficient `0.001460` (raises CT win probability)
- `lag_10__T_A_site_active_smokes`: coefficient `0.001374` (raises CT win probability)
- `lag_06__T_flash_duration_sum`: coefficient `0.001326` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `-0.001278` (lowers CT win probability)
- `lag_08__T_B_site_active_smokes`: coefficient `0.001262` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_13__T_place_HUT`: coefficient `0.004181` (raises CT win probability)
- `lag_08__CT_place_RAFTERS`: coefficient `-0.003489` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003033` (raises CT win probability)
- `lag_14__T_damage_last_5s`: coefficient `0.002973` (raises CT win probability)
- `lag_13__CT_place_HELL`: coefficient `-0.002958` (lowers CT win probability)
- `lag_00__CT5__alive`: coefficient `0.002732` (raises CT win probability)
- `lag_12__T_place_HUT`: coefficient `0.002717` (raises CT win probability)
- `lag_02__CT_place_HUTROOF`: coefficient `-0.002659` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002571` (raises CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `0.002546` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `8283`, seconds `16.50`, LSTM delta `+0.3696`

Top all feature movements:
- `lag_00__T_place_HUT`: contribution `+0.022909`
- `lag_11__T_place_HUT`: contribution `+0.021982`
- `lag_02__CT_place_HUT`: contribution `+0.020135`
- `lag_02__CT_place_HUTROOF`: contribution `+0.018603`
- `lag_05__T2__flash_duration`: contribution `+0.013794`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.013794`
- `lag_06__T2__flash_duration`: contribution `+0.012002`
- `lag_06__T_flash_duration_sum`: contribution `+0.007910`

### tick `9595`, seconds `37.00`, LSTM delta `-0.2364`

Top all feature movements:
- `lag_13__T_place_HUT`: contribution `-0.038976`
- `lag_00__T_place_VENTS`: contribution `-0.023979`
- `lag_15__T_place_HUT`: contribution `-0.017057`
- `lag_05__T_place_SQUEAKY`: contribution `-0.008579`
- `lag_00__T_kills_last_3s`: contribution `-0.007829`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `8123`, seconds `14.00`, LSTM delta `+0.2147`

Top all feature movements:
- `lag_14__CT_place_HELL`: contribution `+0.024088`
- `lag_08__CT_place_RAFTERS`: contribution `+0.018643`
- `lag_13__CT_place_HELL`: contribution `+0.016039`
- `lag_01__T2__flash_duration`: contribution `+0.012287`
- `lag_11__CT_place_HEAVEN`: contribution `+0.010791`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.012287`
- `lag_00__T2__flash_duration`: contribution `+0.009213`

### tick `7995`, seconds `12.00`, LSTM delta `-0.1896`

Top all feature movements:
- `lag_13__CT_place_HELL`: contribution `-0.048118`
- `lag_08__CT_place_RAFTERS`: contribution `-0.018643`
- `lag_14__CT_place_HELL`: contribution `-0.012044`
- `lag_11__CT_place_HEAVEN`: contribution `-0.010791`
- `lag_08__CT_place_HEAVEN`: contribution `-0.010462`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.004648`
- `lag_01__CT4__flash_duration`: contribution `-0.004541`

### tick `8027`, seconds `12.50`, LSTM delta `-0.1856`

Top all feature movements:
- `lag_14__CT_place_HELL`: contribution `-0.036133`
- `lag_08__CT_place_RAFTERS`: contribution `-0.018643`
- `lag_02__CT_place_HUTROOF`: contribution `-0.018603`
- `lag_05__T_place_SQUEAKY`: contribution `+0.017158`
- `lag_11__CT_place_HEAVEN`: contribution `-0.010791`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.003824`
- `lag_03__CT4__flash_duration`: contribution `-0.003711`
