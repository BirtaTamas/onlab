# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `2`

## Largest probability jumps

- tick `6916`, seconds `0.50`, LSTM `0.0203`, delta `-0.0307`
- tick `9444`, seconds `40.00`, LSTM `0.0274`, delta `+0.0177`
- tick `9604`, seconds `42.50`, LSTM `0.0187`, delta `-0.0155`
- tick `9572`, seconds `42.00`, LSTM `0.0342`, delta `+0.0098`
- tick `9476`, seconds `40.50`, LSTM `0.0219`, delta `-0.0055`
- tick `9796`, seconds `45.50`, LSTM `0.0211`, delta `+0.0054`
- tick `7300`, seconds `6.50`, LSTM `0.0205`, delta `+0.0053`
- tick `9060`, seconds `34.00`, LSTM `0.0077`, delta `-0.0052`
- tick `7460`, seconds `9.00`, LSTM `0.0165`, delta `-0.0044`
- tick `10116`, seconds `50.50`, LSTM `0.0348`, delta `+0.0042`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000569`, |coef| `0.000569`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000206`, |coef| `0.000206`
- `lag_05__CT_place_CTSIDEUPPER`: coefficient `-0.000198`, |coef| `0.000198`
- `lag_14__T_place_MAIN`: coefficient `-0.000191`, |coef| `0.000191`
- `lag_06__CT_place_CTSIDEUPPER`: coefficient `-0.000178`, |coef| `0.000178`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000138`, |coef| `0.000138`
- `lag_11__T_kills_last_3s`: coefficient `0.000135`, |coef| `0.000135`
- `lag_15__T_shots_fired_sum`: coefficient `0.000129`, |coef| `0.000129`
- `lag_13__CT_place_MAIN`: coefficient `0.000129`, |coef| `0.000129`
- `lag_12__CT_place_MAIN`: coefficient `-0.000124`, |coef| `0.000124`
- `lag_00__CT_velocity_mean`: coefficient `-0.000122`, |coef| `0.000122`
- `lag_00__T_velocity_mean`: coefficient `-0.000110`, |coef| `0.000110`
- `lag_10__T_place_MAIN`: coefficient `-0.000104`, |coef| `0.000104`
- `lag_01__T_place_MAIN`: coefficient `0.000101`, |coef| `0.000101`
- `lag_01__CT_closest_enemy_dist`: coefficient `-0.000100`, |coef| `0.000100`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000100` (raises CT win probability)
- `lag_01__CT_flash_alpha_mean`: coefficient `0.000099` (raises CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000096` (raises CT win probability)
- `lag_12__T_utility_damage_last_5s`: coefficient `-0.000087` (lowers CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000086` (raises CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000074` (lowers CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000072` (lowers CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000069` (lowers CT win probability)
- `lag_01__T_molly_inv`: coefficient `-0.000067` (lowers CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000067` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.000569` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.000206` (raises CT win probability)
- `lag_05__CT_place_CTSIDEUPPER`: coefficient `-0.000198` (lowers CT win probability)
- `lag_14__T_place_MAIN`: coefficient `-0.000191` (lowers CT win probability)
- `lag_06__CT_place_CTSIDEUPPER`: coefficient `-0.000178` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000138` (lowers CT win probability)
- `lag_11__T_kills_last_3s`: coefficient `0.000135` (raises CT win probability)
- `lag_15__T_shots_fired_sum`: coefficient `0.000129` (raises CT win probability)
- `lag_13__CT_place_MAIN`: coefficient `0.000129` (raises CT win probability)
- `lag_12__CT_place_MAIN`: coefficient `-0.000124` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `6916`, seconds `0.50`, LSTM delta `-0.0307`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.014654`
- `lag_01__T_place_TSPAWN`: contribution `-0.000612`
- `lag_00__T_velocity_mean`: contribution `-0.000393`
- `lag_00__CT_velocity_mean`: contribution `-0.000386`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.000331`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.000307`
- `lag_01__smoke_inv_diff`: contribution `-0.000305`
- `lag_01__CT_flash_alpha_mean`: contribution `-0.000281`
- `lag_01__molly_inv_diff`: contribution `-0.000240`
- `lag_01__T5__flash`: contribution `-0.000155`

### tick `9444`, seconds `40.00`, LSTM delta `+0.0177`

Top all feature movements:
- `lag_12__CT_place_MAIN`: contribution `+0.001671`
- `lag_10__T_shots_fired_sum`: contribution `+0.000828`
- `lag_10__T_place_MAIN`: contribution `+0.000674`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.000661`
- `lag_00__T_place_MAIN`: contribution `+0.000628`

Top utility-only movements:
- `lag_12__T_utility_damage_last_5s`: contribution `+0.000661`
- `lag_12__utility_damage_diff_last_5s`: contribution `+0.000267`

### tick `9604`, seconds `42.50`, LSTM delta `-0.0155`

Top all feature movements:
- `lag_15__T_shots_fired_sum`: contribution `-0.001357`
- `lag_14__T_place_MAIN`: contribution `-0.001232`
- `lag_13__CT_place_MAIN`: contribution `-0.000868`
- `lag_11__T_kills_last_3s`: contribution `-0.000857`
- `lag_01__T_place_MAIN`: contribution `-0.000651`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9572`, seconds `42.00`, LSTM delta `+0.0098`

Top all feature movements:
- `lag_14__T_place_MAIN`: contribution `+0.001232`
- `lag_12__CT_place_MAIN`: contribution `+0.000836`
- `lag_15__T_shots_fired_sum`: contribution `+0.000678`
- `lag_00__T_place_MAIN`: contribution `+0.000628`
- `lag_14__T_shots_fired_sum`: contribution `+0.000411`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9476`, seconds `40.50`, LSTM delta `-0.0055`

Top all feature movements:
- `lag_13__CT_place_MAIN`: contribution `-0.001736`
- `lag_12__CT_place_MAIN`: contribution `+0.000836`
- `lag_10__T_place_MAIN`: contribution `-0.000674`
- `lag_01__T_place_MAIN`: contribution `-0.000651`
- `lag_11__T_shots_fired_sum`: contribution `-0.000598`

Top utility-only movements:
- `lag_13__T_utility_damage_last_5s`: contribution `+0.000204`
- `lag_12__T_utility_damage_last_5s`: contribution `+0.000200`
