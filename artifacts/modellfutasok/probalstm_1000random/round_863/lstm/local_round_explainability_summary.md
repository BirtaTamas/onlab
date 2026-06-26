# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m1-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `121841`, seconds `36.00`, LSTM `0.2869`, delta `-0.3167`
- tick `121105`, seconds `24.50`, LSTM `0.8030`, delta `+0.2381`
- tick `121745`, seconds `34.50`, LSTM `0.6028`, delta `-0.2186`
- tick `120625`, seconds `17.00`, LSTM `0.5240`, delta `-0.1945`
- tick `121201`, seconds `26.00`, LSTM `0.7672`, delta `-0.1635`
- tick `121873`, seconds `36.50`, LSTM `0.1257`, delta `-0.1612`
- tick `121521`, seconds `31.00`, LSTM `0.7675`, delta `+0.1337`
- tick `120497`, seconds `15.00`, LSTM `0.6661`, delta `+0.1181`
- tick `121169`, seconds `25.50`, LSTM `0.9307`, delta `+0.0988`
- tick `124689`, seconds `80.50`, LSTM `0.1664`, delta `+0.0709`

## Top 15 local ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.007078`, |coef| `0.007078`
- `lag_00__kill_diff_last_3s`: coefficient `0.003121`, |coef| `0.003121`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003043`, |coef| `0.003043`
- `lag_00__T_kills_last_3s`: coefficient `-0.002573`, |coef| `0.002573`
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.002414`, |coef| `0.002414`
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.002259`, |coef| `0.002259`
- `lag_09__T_shots_fired_sum`: coefficient `0.002183`, |coef| `0.002183`
- `lag_06__T_shots_fired_sum`: coefficient `-0.002182`, |coef| `0.002182`
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.002128`, |coef| `0.002128`
- `lag_09__CT3__duck_amount`: coefficient `0.002027`, |coef| `0.002027`
- `lag_00__damage_diff_last_5s`: coefficient `0.002025`, |coef| `0.002025`
- `lag_06__CT_place_MIDDLE`: coefficient `0.001965`, |coef| `0.001965`
- `lag_00__CT_defusing_count`: coefficient `0.001836`, |coef| `0.001836`
- `lag_00__T_damage_last_5s`: coefficient `-0.001826`, |coef| `0.001826`
- `lag_09__T2__shots_fired`: coefficient `0.001808`, |coef| `0.001808`

## Top 10 utility ridge features

- `lag_05__T_utility_damage_last_5s`: coefficient `0.001686` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.001543` (raises CT win probability)
- `lag_05__utility_damage_diff_last_5s`: coefficient `-0.001478` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.001417` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.001344` (raises CT win probability)
- `lag_05__T_active_infernos`: coefficient `0.001152` (raises CT win probability)
- `lag_00__CT1__smoke`: coefficient `0.001127` (raises CT win probability)
- `lag_10__CT_active_smokes`: coefficient `0.001100` (raises CT win probability)
- `lag_04__CT3__smoke`: coefficient `0.001056` (raises CT win probability)
- `lag_03__CT3__smoke`: coefficient `0.001054` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.007078` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003121` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003043` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002573` (lowers CT win probability)
- `lag_06__CT_place_TSIDEUPPER`: coefficient `-0.002414` (lowers CT win probability)
- `lag_10__T_place_SIDEENTRANCE`: coefficient `-0.002259` (lowers CT win probability)
- `lag_09__T_shots_fired_sum`: coefficient `0.002183` (raises CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `-0.002182` (lowers CT win probability)
- `lag_07__CT_place_TSIDEUPPER`: coefficient `-0.002128` (lowers CT win probability)
- `lag_09__CT3__duck_amount`: coefficient `0.002027` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121841`, seconds `36.00`, LSTM delta `-0.3167`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.053204`
- `lag_06__CT_place_TSIDEUPPER`: contribution `-0.018148`
- `lag_10__T_place_SIDEENTRANCE`: contribution `-0.011024`
- `lag_00__T_kills_last_3s`: contribution `-0.008152`
- `lag_00__kill_diff_last_3s`: contribution `-0.007511`

Top utility-only movements:
- `lag_05__T_B_site_active_infernos`: contribution `-0.004363`

### tick `121105`, seconds `24.50`, LSTM delta `+0.2381`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.039255`
- `lag_06__T2__shots_fired`: contribution `+0.024350`
- `lag_00__CT_shots_fired_sum`: contribution `+0.014797`
- `lag_09__T_shots_fired_sum`: contribution `+0.008184`
- `lag_00__kill_diff_last_3s`: contribution `+0.007511`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.005682`
- `lag_02__T1__flash_duration`: contribution `+0.005673`
- `lag_00__T5__flash_duration`: contribution `+0.005512`
- `lag_02__T_flash_duration_sum`: contribution `+0.004495`

### tick `121745`, seconds `34.50`, LSTM delta `-0.2186`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.053204`
- `lag_00__T_kills_last_3s`: contribution `-0.008152`
- `lag_00__kill_diff_last_3s`: contribution `-0.007511`
- `lag_07__T_place_SIDEENTRANCE`: contribution `-0.006439`
- `lag_10__T_place_TSIDELOWER`: contribution `-0.006393`

Top utility-only movements:
- `lag_02__T_B_site_active_infernos`: contribution `-0.003801`
- `lag_00__CT3__smoke`: contribution `-0.003135`

### tick `120625`, seconds `17.00`, LSTM delta `-0.1945`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.053204`
- `lag_00__T_kills_last_3s`: contribution `-0.008152`
- `lag_00__kill_diff_last_3s`: contribution `-0.007511`
- `lag_04__T_flashed_players`: contribution `-0.005974`
- `lag_04__T1__duck_amount`: contribution `-0.005748`

Top utility-only movements:
- `lag_05__T_B_site_active_infernos`: contribution `-0.004363`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.002879`
- `lag_05__active_infernos_total`: contribution `-0.002564`
- `lag_05__T_active_infernos`: contribution `-0.002400`

### tick `121201`, seconds `26.00`, LSTM delta `-0.1635`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.048620`
- `lag_09__T_shots_fired_sum`: contribution `-0.039284`
- `lag_09__T2__shots_fired`: contribution `-0.025525`
- `lag_00__T_kills_last_3s`: contribution `-0.008152`
- `lag_00__kill_diff_last_3s`: contribution `-0.007511`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.003713`
- `lag_01__T1__flash_duration`: contribution `-0.003297`
