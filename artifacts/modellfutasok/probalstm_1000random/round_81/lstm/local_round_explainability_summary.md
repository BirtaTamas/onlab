# Local Round Explainability

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `16`

## Largest probability jumps

- tick `147238`, seconds `74.50`, LSTM `0.8726`, delta `+0.1706`
- tick `143782`, seconds `20.50`, LSTM `0.7067`, delta `+0.1545`
- tick `148102`, seconds `88.00`, LSTM `0.9659`, delta `+0.0675`
- tick `143270`, seconds `12.50`, LSTM `0.5646`, delta `-0.0348`
- tick `143462`, seconds `15.50`, LSTM `0.5498`, delta `+0.0318`
- tick `146790`, seconds `67.50`, LSTM `0.7435`, delta `+0.0292`
- tick `143174`, seconds `11.00`, LSTM `0.5966`, delta `-0.0291`
- tick `144742`, seconds `35.50`, LSTM `0.6604`, delta `-0.0291`
- tick `144294`, seconds `28.50`, LSTM `0.6708`, delta `+0.0288`
- tick `145158`, seconds `42.00`, LSTM `0.6364`, delta `-0.0285`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001391`, |coef| `0.001391`
- `lag_10__CT_place_JUNGLE`: coefficient `0.001381`, |coef| `0.001381`
- `lag_01__CT_place_TRUCK`: coefficient `0.001358`, |coef| `0.001358`
- `lag_11__T_shots_fired_sum`: coefficient `-0.001331`, |coef| `0.001331`
- `lag_07__CT_place_SNIPERSNEST`: coefficient `-0.001300`, |coef| `0.001300`
- `lag_01__T_place_CONNECTOR`: coefficient `0.001261`, |coef| `0.001261`
- `lag_00__kill_diff_last_3s`: coefficient `0.001160`, |coef| `0.001160`
- `lag_00__CT_damage_last_5s`: coefficient `0.001156`, |coef| `0.001156`
- `lag_07__CT2__duck_amount`: coefficient `0.001140`, |coef| `0.001140`
- `lag_11__CT2__duck_amount`: coefficient `-0.001104`, |coef| `0.001104`
- `lag_00__damage_diff_last_5s`: coefficient `0.001023`, |coef| `0.001023`
- `lag_02__T1__flash_duration`: coefficient `0.001002`, |coef| `0.001002`
- `lag_00__T2__utility_total`: coefficient `-0.000948`, |coef| `0.000948`
- `lag_06__CT_place_JUNGLE`: coefficient `0.000920`, |coef| `0.000920`
- `lag_14__T_place_STAIRS`: coefficient `0.000888`, |coef| `0.000888`

## Top 10 utility ridge features

- `lag_02__T1__flash_duration`: coefficient `0.001002` (raises CT win probability)
- `lag_00__T2__utility_total`: coefficient `-0.000948` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000851` (lowers CT win probability)
- `lag_05__active_infernos_total`: coefficient `0.000686` (raises CT win probability)
- `lag_02__T_flash_duration_sum`: coefficient `0.000668` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.000663` (raises CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000644` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000635` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `0.000630` (raises CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `0.000586` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001391` (raises CT win probability)
- `lag_10__CT_place_JUNGLE`: coefficient `0.001381` (raises CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `0.001358` (raises CT win probability)
- `lag_11__T_shots_fired_sum`: coefficient `-0.001331` (lowers CT win probability)
- `lag_07__CT_place_SNIPERSNEST`: coefficient `-0.001300` (lowers CT win probability)
- `lag_01__T_place_CONNECTOR`: coefficient `0.001261` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001160` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001156` (raises CT win probability)
- `lag_07__CT2__duck_amount`: coefficient `0.001140` (raises CT win probability)
- `lag_11__CT2__duck_amount`: coefficient `-0.001104` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `147238`, seconds `74.50`, LSTM delta `+0.1706`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `+0.010981`
- `lag_07__CT_place_SNIPERSNEST`: contribution `+0.006963`
- `lag_01__T_place_CONNECTOR`: contribution `+0.006108`
- `lag_06__CT_place_JUNGLE`: contribution `+0.005904`
- `lag_02__T_flashed_players`: contribution `+0.005127`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `+0.004548`
- `lag_02__T_flash_duration_sum`: contribution `+0.002577`

### tick `143782`, seconds `20.50`, LSTM delta `+0.1545`

Top all feature movements:
- `lag_10__CT_place_JUNGLE`: contribution `+0.008862`
- `lag_01__CT_place_TRUCK`: contribution `+0.008757`
- `lag_05__CT_place_SNIPERSNEST`: contribution `+0.004277`
- `lag_14__CT_place_SNIPERSNEST`: contribution `+0.004228`
- `lag_00__CT_kills_last_3s`: contribution `+0.004016`

Top utility-only movements:
- `lag_00__T2__utility_total`: contribution `+0.003111`
- `lag_00__T2__flash`: contribution `+0.002505`

### tick `148102`, seconds `88.00`, LSTM delta `+0.0675`

Top all feature movements:
- `lag_00__T_place_STAIRS`: contribution `+0.016424`
- `lag_09__T_place_STAIRS`: contribution `+0.016199`
- `lag_07__CT2__duck_amount`: contribution `+0.004342`
- `lag_00__CT_kills_last_3s`: contribution `+0.004016`
- `lag_02__CT1__flash_duration`: contribution `+0.003460`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `+0.003460`

### tick `143270`, seconds `12.50`, LSTM delta `-0.0348`

Top all feature movements:
- `lag_00__CT_place_SNIPERSNEST`: contribution `-0.004223`
- `lag_10__CT_place_SHOP`: contribution `-0.003882`
- `lag_00__CT1__is_scoped`: contribution `-0.003641`
- `lag_10__CT_place_SNIPERSNEST`: contribution `-0.003080`
- `lag_09__CT1__duck_amount`: contribution `+0.002968`

Top utility-only movements:
- `lag_10__CT2__flash_duration`: contribution `-0.001659`
- `lag_02__CT2__flash_duration`: contribution `-0.001403`
- `lag_05__T1__flash_duration`: contribution `-0.001099`

### tick `143462`, seconds `15.50`, LSTM delta `+0.0318`

Top all feature movements:
- `lag_00__CT_place_JUNGLE`: contribution `+0.005531`
- `lag_09__CT1__duck_amount`: contribution `-0.002968`
- `lag_15__CT1__duck_amount`: contribution `+0.002907`
- `lag_00__CT2__duck_amount`: contribution `+0.002865`
- `lag_00__CT4__duck_amount`: contribution `+0.002626`

Top utility-only movements:
- No utility movement among the top local contributors.
