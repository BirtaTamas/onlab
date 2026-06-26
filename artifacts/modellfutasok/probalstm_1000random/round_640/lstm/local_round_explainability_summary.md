# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `2`

## Largest probability jumps

- tick `14582`, seconds `79.50`, LSTM `0.0543`, delta `-0.3018`
- tick `13270`, seconds `59.00`, LSTM `0.3734`, delta `+0.1993`
- tick `10934`, seconds `22.50`, LSTM `0.0595`, delta `-0.1060`
- tick `9526`, seconds `0.50`, LSTM `0.1649`, delta `-0.0768`
- tick `13302`, seconds `59.50`, LSTM `0.4289`, delta `+0.0555`
- tick `9814`, seconds `5.00`, LSTM `0.2719`, delta `+0.0493`
- tick `9846`, seconds `5.50`, LSTM `0.3135`, delta `+0.0416`
- tick `14614`, seconds `80.00`, LSTM `0.0183`, delta `-0.0360`
- tick `14230`, seconds `74.00`, LSTM `0.4042`, delta `-0.0357`
- tick `13334`, seconds `60.00`, LSTM `0.4629`, delta `+0.0341`

## Top 15 local ridge features

- `lag_15__T_place_ARCH`: coefficient `-0.007471`, |coef| `0.007471`
- `lag_00__kill_diff_last_3s`: coefficient `0.003584`, |coef| `0.003584`
- `lag_00__CT_place_APARTMENTS`: coefficient `0.003297`, |coef| `0.003297`
- `lag_00__T_kills_last_3s`: coefficient `-0.002989`, |coef| `0.002989`
- `lag_08__CT_place_BALCONY`: coefficient `-0.002869`, |coef| `0.002869`
- `lag_07__T2__duck_amount`: coefficient `0.002545`, |coef| `0.002545`
- `lag_00__damage_diff_last_5s`: coefficient `0.002411`, |coef| `0.002411`
- `lag_11__T3__duck_amount`: coefficient `-0.002399`, |coef| `0.002399`
- `lag_00__CT4__alive`: coefficient `0.002279`, |coef| `0.002279`
- `lag_00__T_damage_last_5s`: coefficient `-0.002245`, |coef| `0.002245`
- `lag_00__CT4__hp`: coefficient `0.002245`, |coef| `0.002245`
- `lag_01__T2__is_walking`: coefficient `0.002222`, |coef| `0.002222`
- `lag_00__T4__flash_duration`: coefficient `-0.002202`, |coef| `0.002202`
- `lag_14__T_place_ARCH`: coefficient `-0.002163`, |coef| `0.002163`
- `lag_13__T_place_ARCH`: coefficient `-0.002151`, |coef| `0.002151`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `-0.002202` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.002027` (raises CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `-0.001895` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001639` (raises CT win probability)
- `lag_09__CT5__flash_duration`: coefficient `0.001492` (raises CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `0.001450` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.001310` (lowers CT win probability)
- `lag_06__T_B_site_active_smokes`: coefficient `0.001228` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.001198` (raises CT win probability)
- `lag_09__T_B_site_active_smokes`: coefficient `0.001169` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_ARCH`: coefficient `-0.007471` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003584` (raises CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `0.003297` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002989` (lowers CT win probability)
- `lag_08__CT_place_BALCONY`: coefficient `-0.002869` (lowers CT win probability)
- `lag_07__T2__duck_amount`: coefficient `0.002545` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002411` (raises CT win probability)
- `lag_11__T3__duck_amount`: coefficient `-0.002399` (lowers CT win probability)
- `lag_00__CT4__alive`: coefficient `0.002279` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002245` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14582`, seconds `79.50`, LSTM delta `-0.3018`

Top all feature movements:
- `lag_15__T_place_ARCH`: contribution `-0.069505`
- `lag_00__CT_place_APARTMENTS`: contribution `-0.012666`
- `lag_00__T_kills_last_3s`: contribution `-0.009470`
- `lag_11__T3__duck_amount`: contribution `-0.008797`
- `lag_07__T2__duck_amount`: contribution `-0.008665`

Top utility-only movements:
- `lag_00__CT4__smoke`: contribution `-0.004425`

### tick `13270`, seconds `59.00`, LSTM delta `+0.1993`

Top all feature movements:
- `lag_08__CT_place_BALCONY`: contribution `+0.018411`
- `lag_00__T4__flash_duration`: contribution `+0.014941`
- `lag_11__CT_place_BALCONY`: contribution `+0.011654`
- `lag_01__T4__flash_duration`: contribution `+0.011118`
- `lag_01__CT5__flash_duration`: contribution `+0.009538`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `+0.014941`
- `lag_01__T4__flash_duration`: contribution `+0.011118`
- `lag_01__CT5__flash_duration`: contribution `+0.009538`

### tick `10934`, seconds `22.50`, LSTM delta `-0.1060`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009470`
- `lag_00__kill_diff_last_3s`: contribution `-0.008627`
- `lag_02__CT_place_BALCONY`: contribution `-0.008353`
- `lag_08__CT1__flash_duration`: contribution `-0.006092`
- `lag_00__damage_diff_last_5s`: contribution `-0.005438`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `-0.006092`
- `lag_00__CT1__flash_duration`: contribution `-0.003006`

### tick `9526`, seconds `0.50`, LSTM delta `-0.0768`

Top all feature movements:
- `lag_01__T_place_TSPAWN`: contribution `-0.004829`
- `lag_01__T_closest_enemy_dist`: contribution `-0.004349`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.004115`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003921`
- `lag_01__centroid_distance_xy`: contribution `-0.003919`

Top utility-only movements:
- `lag_01__T4__smoke`: contribution `-0.001756`
- `lag_01__T_smoke_inv`: contribution `-0.001477`
- `lag_01__CT2__smoke`: contribution `-0.001032`
- `lag_01__T3__flash`: contribution `-0.000941`
- `lag_01__T5__flash`: contribution `-0.000887`

### tick `13302`, seconds `59.50`, LSTM delta `+0.0555`

Top all feature movements:
- `lag_11__CT_place_BALCONY`: contribution `-0.011654`
- `lag_01__T4__flash_duration`: contribution `-0.011118`
- `lag_12__CT_place_BALCONY`: contribution `+0.009712`
- `lag_09__CT_place_BALCONY`: contribution `+0.007044`
- `lag_02__T2__duck_amount`: contribution `-0.006377`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `-0.011118`
- `lag_02__T4__flash_duration`: contribution `+0.005716`
- `lag_01__T4__smoke`: contribution `+0.002522`
- `lag_02__CT5__flash_duration`: contribution `+0.002476`
