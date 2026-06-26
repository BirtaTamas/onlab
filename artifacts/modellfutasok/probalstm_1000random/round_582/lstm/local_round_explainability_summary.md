# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `8`

## Largest probability jumps

- tick `76244`, seconds `0.50`, LSTM `0.1975`, delta `-0.0678`
- tick `78740`, seconds `39.50`, LSTM `0.2025`, delta `-0.0567`
- tick `78164`, seconds `30.50`, LSTM `0.1977`, delta `-0.0424`
- tick `78260`, seconds `32.00`, LSTM `0.2270`, delta `+0.0393`
- tick `78772`, seconds `40.00`, LSTM `0.1636`, delta `-0.0389`
- tick `78356`, seconds `33.50`, LSTM `0.2093`, delta `-0.0327`
- tick `77108`, seconds `14.00`, LSTM `0.2307`, delta `+0.0312`
- tick `77300`, seconds `17.00`, LSTM `0.2628`, delta `+0.0284`
- tick `78580`, seconds `37.00`, LSTM `0.2596`, delta `+0.0280`
- tick `78292`, seconds `32.50`, LSTM `0.2550`, delta `+0.0280`

## Top 15 local ridge features

- `lag_00__T_place_SIDEHALL`: coefficient `-0.001254`, |coef| `0.001254`
- `lag_00__CT1__is_walking`: coefficient `0.001094`, |coef| `0.001094`
- `lag_00__T_place_CTSPAWN`: coefficient `-0.000972`, |coef| `0.000972`
- `lag_01__T_place_SIDEHALL`: coefficient `-0.000834`, |coef| `0.000834`
- `lag_01__CT1__is_walking`: coefficient `0.000805`, |coef| `0.000805`
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.000801`, |coef| `0.000801`
- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.000756`, |coef| `0.000756`
- `lag_00__CT_velocity_mean`: coefficient `-0.000729`, |coef| `0.000729`
- `lag_01__T1__is_walking`: coefficient `0.000704`, |coef| `0.000704`
- `lag_00__T_velocity_mean`: coefficient `-0.000681`, |coef| `0.000681`
- `lag_01__T_place_CTSPAWN`: coefficient `-0.000680`, |coef| `0.000680`
- `lag_08__T_place_TSIDEUPPER`: coefficient `0.000679`, |coef| `0.000679`
- `lag_01__CT_walking_count`: coefficient `0.000679`, |coef| `0.000679`
- `lag_08__CT5__flash_duration`: coefficient `-0.000671`, |coef| `0.000671`
- `lag_00__T_place_MIDDLE`: coefficient `0.000670`, |coef| `0.000670`

## Top 10 utility ridge features

- `lag_08__CT5__flash_duration`: coefficient `-0.000671` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.000552` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000447` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.000428` (raises CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.000416` (raises CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `0.000413` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `0.000400` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.000399` (lowers CT win probability)
- `lag_14__T_utility_damage_last_5s`: coefficient `0.000383` (raises CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `0.000380` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEHALL`: coefficient `-0.001254` (lowers CT win probability)
- `lag_00__CT1__is_walking`: coefficient `0.001094` (raises CT win probability)
- `lag_00__T_place_CTSPAWN`: coefficient `-0.000972` (lowers CT win probability)
- `lag_01__T_place_SIDEHALL`: coefficient `-0.000834` (lowers CT win probability)
- `lag_01__CT1__is_walking`: coefficient `0.000805` (raises CT win probability)
- `lag_01__CT_place_SIDEENTRANCE`: coefficient `-0.000801` (lowers CT win probability)
- `lag_01__CT_place_TSIDEUPPER`: coefficient `-0.000756` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000729` (lowers CT win probability)
- `lag_01__T1__is_walking`: coefficient `0.000704` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000681` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `76244`, seconds `0.50`, LSTM delta `-0.0678`

Top all feature movements:
- `lag_01__CT_place_TSIDEUPPER`: contribution `-0.005600`
- `lag_01__CT_place_WATER`: contribution `-0.003441`
- `lag_01__CT_place_SIDEENTRANCE`: contribution `-0.003070`
- `lag_00__CT_velocity_mean`: contribution `-0.002507`
- `lag_00__T_velocity_mean`: contribution `-0.002142`

Top utility-only movements:
- `lag_01__smoke_inv_diff`: contribution `-0.000954`
- `lag_01__T_smoke_inv`: contribution `-0.000441`
- `lag_01__molly_inv_diff`: contribution `-0.000318`
- `lag_01__utility_inv_diff`: contribution `-0.000306`
- `lag_01__T3__utility_total`: contribution `-0.000277`

### tick `78740`, seconds `39.50`, LSTM delta `-0.0567`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `-0.008128`
- `lag_08__CT5__flash_duration`: contribution `-0.005089`
- `lag_08__T_place_TSIDEUPPER`: contribution `-0.001714`
- `lag_11__T_B_site_active_infernos`: contribution `-0.001561`
- `lag_03__T3__is_walking`: contribution `-0.001539`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `-0.005089`
- `lag_11__T_B_site_active_infernos`: contribution `-0.001561`
- `lag_00__T_A_site_active_infernos`: contribution `-0.001192`
- `lag_08__CT_flash_duration_sum`: contribution `-0.001036`

### tick `78164`, seconds `30.50`, LSTM delta `-0.0424`

Top all feature movements:
- `lag_00__CT1__is_walking`: contribution `-0.002554`
- `lag_00__T5__duck_amount`: contribution `-0.001934`
- `lag_03__T2__duck_amount`: contribution `-0.001488`
- `lag_02__T4__is_walking`: contribution `-0.001450`
- `lag_08__T1__duck_amount`: contribution `-0.001043`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `-0.000837`
- `lag_15__T3__smoke`: contribution `-0.000688`

### tick `78260`, seconds `32.00`, LSTM delta `+0.0393`

Top all feature movements:
- `lag_00__CT1__is_walking`: contribution `+0.002554`
- `lag_00__T5__duck_amount`: contribution `+0.001934`
- `lag_01__T1__is_walking`: contribution `+0.001605`
- `lag_09__T1__duck_amount`: contribution `+0.001454`
- `lag_02__T4__is_walking`: contribution `+0.001450`

Top utility-only movements:
- `lag_10__T_B_site_active_infernos`: contribution `+0.001209`

### tick `78772`, seconds `40.00`, LSTM delta `-0.0389`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `-0.008128`
- `lag_01__T_place_SIDEHALL`: contribution `-0.005408`
- `lag_09__CT5__flash_duration`: contribution `-0.001674`
- `lag_01__T1__is_walking`: contribution `-0.001605`
- `lag_03__T3__is_walking`: contribution `+0.001539`

Top utility-only movements:
- `lag_09__CT5__flash_duration`: contribution `-0.001674`
- `lag_12__T_B_site_active_infernos`: contribution `-0.001264`
- `lag_01__T_A_site_active_infernos`: contribution `-0.000734`
