# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `13233`, seconds `64.50`, LSTM `0.1759`, delta `-0.1048`
- tick `9137`, seconds `0.50`, LSTM `0.1766`, delta `-0.0591`
- tick `13297`, seconds `65.50`, LSTM `0.0912`, delta `-0.0581`
- tick `10577`, seconds `23.00`, LSTM `0.2731`, delta `+0.0546`
- tick `12753`, seconds `57.00`, LSTM `0.2162`, delta `-0.0444`
- tick `12241`, seconds `49.00`, LSTM `0.2477`, delta `+0.0379`
- tick `10801`, seconds `26.50`, LSTM `0.2078`, delta `+0.0376`
- tick `13425`, seconds `67.50`, LSTM `0.0270`, delta `-0.0368`
- tick `9681`, seconds `9.00`, LSTM `0.1776`, delta `+0.0356`
- tick `10673`, seconds `24.50`, LSTM `0.2118`, delta `-0.0341`

## Top 15 local ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.002133`, |coef| `0.002133`
- `lag_15__T_place_LADDER`: coefficient `-0.001698`, |coef| `0.001698`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001489`, |coef| `0.001489`
- `lag_00__CT3__is_scoped`: coefficient `0.000924`, |coef| `0.000924`
- `lag_01__CT1__is_walking`: coefficient `0.000863`, |coef| `0.000863`
- `lag_08__CT2__duck_amount`: coefficient `0.000853`, |coef| `0.000853`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000833`, |coef| `0.000833`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000800`, |coef| `0.000800`
- `lag_00__CT_velocity_mean`: coefficient `-0.000787`, |coef| `0.000787`
- `lag_01__CT_place_TRUCK`: coefficient `0.000782`, |coef| `0.000782`
- `lag_05__T_place_LADDER`: coefficient `0.000758`, |coef| `0.000758`
- `lag_00__CT2__duck_amount`: coefficient `-0.000737`, |coef| `0.000737`
- `lag_01__CT_walking_count`: coefficient `0.000728`, |coef| `0.000728`
- `lag_00__T_velocity_mean`: coefficient `-0.000703`, |coef| `0.000703`
- `lag_14__T_place_CONNECTOR`: coefficient `-0.000688`, |coef| `0.000688`

## Top 10 utility ridge features

- `lag_01__T_flash_alpha_mean`: coefficient `0.000502` (raises CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000485` (raises CT win probability)
- `lag_00__T5__smoke`: coefficient `0.000481` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000430` (lowers CT win probability)
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.000421` (lowers CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000419` (raises CT win probability)
- `lag_01__T_active_smokes`: coefficient `0.000406` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000401` (raises CT win probability)
- `lag_01__T5__flash`: coefficient `-0.000398` (lowers CT win probability)
- `lag_01__active_smokes_total`: coefficient `0.000394` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_TRUCK`: coefficient `0.002133` (raises CT win probability)
- `lag_15__T_place_LADDER`: coefficient `-0.001698` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001489` (lowers CT win probability)
- `lag_00__CT3__is_scoped`: coefficient `0.000924` (raises CT win probability)
- `lag_01__CT1__is_walking`: coefficient `0.000863` (raises CT win probability)
- `lag_08__CT2__duck_amount`: coefficient `0.000853` (raises CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000833` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000800` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000787` (lowers CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `0.000782` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `13233`, seconds `64.50`, LSTM delta `-0.1048`

Top all feature movements:
- `lag_15__T_place_LADDER`: contribution `-0.038375`
- `lag_05__T_place_LADDER`: contribution `-0.017144`
- `lag_00__CT_place_TRUCK`: contribution `-0.013761`
- `lag_00__CT3__is_scoped`: contribution `-0.004202`
- `lag_14__T_place_CONNECTOR`: contribution `-0.003332`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9137`, seconds `0.50`, LSTM delta `-0.0591`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.003824`
- `lag_01__T_place_TSPAWN`: contribution `-0.003690`
- `lag_00__CT_velocity_mean`: contribution `-0.002800`
- `lag_00__T_velocity_mean`: contribution `-0.002105`
- `lag_01__T_closest_enemy_dist`: contribution `-0.001429`

Top utility-only movements:
- `lag_01__T_flash_alpha_mean`: contribution `-0.001101`
- `lag_00__T5__smoke`: contribution `-0.001042`
- `lag_01__T_smoke_inv`: contribution `-0.000981`
- `lag_01__molly_inv_diff`: contribution `-0.000875`
- `lag_01__T5__flash`: contribution `-0.000838`

### tick `13297`, seconds `65.50`, LSTM delta `-0.0581`

Top all feature movements:
- `lag_07__T_place_LADDER`: contribution `-0.011346`
- `lag_00__T_shots_fired_sum`: contribution `-0.010048`
- `lag_02__CT3__is_scoped`: contribution `-0.002218`
- `lag_09__CT1__is_walking`: contribution `-0.001563`
- `lag_01__T3__duck_amount`: contribution `-0.001543`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `10577`, seconds `23.00`, LSTM delta `+0.0546`

Top all feature movements:
- `lag_00__CT_place_TRUCK`: contribution `+0.013761`
- `lag_00__T_shots_fired_sum`: contribution `+0.010048`
- `lag_00__T5__shots_fired`: contribution `+0.003600`
- `lag_08__CT2__duck_amount`: contribution `+0.003251`
- `lag_00__CT2__duck_amount`: contribution `+0.002808`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `12753`, seconds `57.00`, LSTM delta `-0.0444`

Top all feature movements:
- `lag_00__T_place_LADDER`: contribution `-0.012030`
- `lag_08__CT2__duck_amount`: contribution `-0.003251`
- `lag_09__CT_place_JUNGLE`: contribution `-0.003025`
- `lag_03__CT_place_TRUCK`: contribution `-0.002445`
- `lag_01__CT1__is_walking`: contribution `-0.002014`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `-0.000867`
