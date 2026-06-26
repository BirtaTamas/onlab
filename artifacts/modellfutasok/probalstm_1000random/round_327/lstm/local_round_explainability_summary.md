# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `10`

## Largest probability jumps

- tick `98041`, seconds `71.50`, LSTM `0.9073`, delta `+0.2802`
- tick `96537`, seconds `48.00`, LSTM `0.1363`, delta `-0.1927`
- tick `94265`, seconds `12.50`, LSTM `0.2977`, delta `-0.1729`
- tick `96505`, seconds `47.50`, LSTM `0.3291`, delta `-0.1474`
- tick `97849`, seconds `68.50`, LSTM `0.6543`, delta `+0.1406`
- tick `95737`, seconds `35.50`, LSTM `0.6405`, delta `+0.1262`
- tick `97209`, seconds `58.50`, LSTM `0.1279`, delta `+0.1135`
- tick `94361`, seconds `14.00`, LSTM `0.4304`, delta `+0.1086`
- tick `98009`, seconds `71.00`, LSTM `0.6271`, delta `+0.1063`
- tick `97561`, seconds `64.00`, LSTM `0.4897`, delta `+0.1028`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003574`, |coef| `0.003574`
- `lag_06__CT_defusing_count`: coefficient `0.003483`, |coef| `0.003483`
- `lag_00__CT_defusing_count`: coefficient `0.003278`, |coef| `0.003278`
- `lag_00__CT_kills_last_3s`: coefficient `0.002895`, |coef| `0.002895`
- `lag_08__CT_place_TSIDEUPPER`: coefficient `0.002604`, |coef| `0.002604`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002577`, |coef| `0.002577`
- `lag_07__CT_place_SIDEHALL`: coefficient `-0.002358`, |coef| `0.002358`
- `lag_00__T_macro_A`: coefficient `-0.002310`, |coef| `0.002310`
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002310`, |coef| `0.002310`
- `lag_05__CT_defusing_count`: coefficient `0.002300`, |coef| `0.002300`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `0.002237`, |coef| `0.002237`
- `lag_13__CT_place_MIDDLE`: coefficient `-0.002229`, |coef| `0.002229`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002181`, |coef| `0.002181`
- `lag_05__T4__is_walking`: coefficient `-0.002016`, |coef| `0.002016`
- `lag_00__CT_place_SIDEHALL`: coefficient `-0.002011`, |coef| `0.002011`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002181` (lowers CT win probability)
- `lag_08__T2__flash_duration`: coefficient `0.001887` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.001836` (lowers CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.001787` (raises CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.001518` (raises CT win probability)
- `lag_10__CT_flashes_last_5s`: coefficient `0.001495` (raises CT win probability)
- `lag_02__T_A_site_active_infernos`: coefficient `0.001487` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.001468` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `-0.001417` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.001411` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003574` (raises CT win probability)
- `lag_06__CT_defusing_count`: coefficient `0.003483` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.003278` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002895` (raises CT win probability)
- `lag_08__CT_place_TSIDEUPPER`: coefficient `0.002604` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002577` (raises CT win probability)
- `lag_07__CT_place_SIDEHALL`: coefficient `-0.002358` (lowers CT win probability)
- `lag_00__T_macro_A`: coefficient `-0.002310` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEA`: coefficient `-0.002310` (lowers CT win probability)
- `lag_05__CT_defusing_count`: coefficient `0.002300` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `98041`, seconds `71.50`, LSTM delta `+0.2802`

Top all feature movements:
- `lag_06__CT_defusing_count`: contribution `+0.033768`
- `lag_00__CT_shots_fired_sum`: contribution `+0.017903`
- `lag_00__kill_diff_last_3s`: contribution `+0.017203`
- `lag_00__CT_kills_last_3s`: contribution `+0.016717`
- `lag_00__T_flash_alpha_mean`: contribution `+0.013235`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.013235`
- `lag_11__CT3__flash_duration`: contribution `+0.012620`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.003799`

### tick `96537`, seconds `48.00`, LSTM delta `-0.1927`

Top all feature movements:
- `lag_08__CT_place_TSIDEUPPER`: contribution `-0.019571`
- `lag_14__CT_place_TSIDEUPPER`: contribution `-0.014962`
- `lag_13__T_bomb_zone_count`: contribution `-0.011222`
- `lag_07__T_bomb_zone_count`: contribution `-0.009933`
- `lag_00__kill_diff_last_3s`: contribution `-0.008601`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `-0.004426`
- `lag_13__T_A_site_active_infernos`: contribution `-0.003766`
- `lag_03__CT4__molly`: contribution `-0.003370`

### tick `94265`, seconds `12.50`, LSTM delta `-0.1729`

Top all feature movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.016438`
- `lag_11__CT_place_TOPOFMID`: contribution `-0.010303`
- `lag_08__CT_place_MIDDLE`: contribution `-0.009023`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.008837`
- `lag_00__kill_diff_last_3s`: contribution `-0.008601`

Top utility-only movements:
- `lag_10__CT_flashes_last_5s`: contribution `-0.016438`
- `lag_00__T3__flash_duration`: contribution `-0.004760`
- `lag_09__T3__flash_duration`: contribution `-0.004735`
- `lag_03__CT5__flash_duration`: contribution `-0.003799`
- `lag_00__CT5__flash_duration`: contribution `-0.003329`

### tick `96505`, seconds `47.50`, LSTM delta `-0.1474`

Top all feature movements:
- `lag_07__CT_place_TSIDEUPPER`: contribution `-0.016812`
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.012347`
- `lag_12__T_bomb_zone_count`: contribution `-0.009325`
- `lag_06__T_bomb_zone_count`: contribution `-0.006557`
- `lag_13__CT_place_MIDDLE`: contribution `-0.005847`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `-0.004426`
- `lag_15__T_A_site_active_infernos`: contribution `-0.004200`
- `lag_01__T_B_site_active_infernos`: contribution `-0.003086`
- `lag_02__CT4__molly`: contribution `-0.002869`
- `lag_12__T_A_site_active_infernos`: contribution `-0.002734`

### tick `97849`, seconds `68.50`, LSTM delta `+0.1406`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.031777`
- `lag_05__CT3__flash_duration`: contribution `+0.009741`
- `lag_02__CT_place_SIDEHALL`: contribution `+0.006202`
- `lag_05__T4__is_walking`: contribution `+0.004654`
- `lag_05__CT_flashed_players`: contribution `+0.003906`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.009741`
- `lag_05__CT_A_site_active_infernos`: contribution `+0.003897`
- `lag_05__CT_flash_duration_sum`: contribution `+0.003132`
- `lag_12__CT4__flash_duration`: contribution `+0.002098`
