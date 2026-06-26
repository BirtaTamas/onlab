# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `14`

## Largest probability jumps

- tick `125972`, seconds `78.50`, LSTM `0.0604`, delta `-0.0944`
- tick `123988`, seconds `47.50`, LSTM `0.2172`, delta `+0.0762`
- tick `124404`, seconds `54.00`, LSTM `0.1733`, delta `-0.0750`
- tick `120980`, seconds `0.50`, LSTM `0.1948`, delta `-0.0580`
- tick `121972`, seconds `16.00`, LSTM `0.2472`, delta `+0.0491`
- tick `124052`, seconds `48.50`, LSTM `0.2767`, delta `+0.0437`
- tick `123348`, seconds `37.50`, LSTM `0.2232`, delta `-0.0368`
- tick `122388`, seconds `22.50`, LSTM `0.2522`, delta `-0.0364`
- tick `123956`, seconds `47.00`, LSTM `0.1410`, delta `-0.0361`
- tick `121940`, seconds `15.50`, LSTM `0.1981`, delta `+0.0358`

## Top 15 local ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.002230`, |coef| `0.002230`
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.001524`, |coef| `0.001524`
- `lag_00__CT2__is_walking`: coefficient `-0.001480`, |coef| `0.001480`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001277`, |coef| `0.001277`
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001070`, |coef| `0.001070`
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001062`, |coef| `0.001062`
- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001049`, |coef| `0.001049`
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.001048`, |coef| `0.001048`
- `lag_02__CT_place_SIDEENTRANCE`: coefficient `0.000991`, |coef| `0.000991`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000938`, |coef| `0.000938`
- `lag_00__T_place_TUNNEL`: coefficient `-0.000930`, |coef| `0.000930`
- `lag_00__T2__is_walking`: coefficient `0.000928`, |coef| `0.000928`
- `lag_12__CT2__is_walking`: coefficient `-0.000901`, |coef| `0.000901`
- `lag_04__T_place_SIDEENTRANCE`: coefficient `-0.000885`, |coef| `0.000885`
- `lag_12__CT5__is_walking`: coefficient `-0.000859`, |coef| `0.000859`

## Top 10 utility ridge features

- `lag_00__T_utility_damage_last_5s`: coefficient `-0.001049` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000664` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000662` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000611` (lowers CT win probability)
- `lag_06__T1__molly`: coefficient `0.000605` (raises CT win probability)
- `lag_14__T5__molly`: coefficient `0.000544` (raises CT win probability)
- `lag_11__CT4__smoke`: coefficient `0.000539` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `-0.000511` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.000480` (lowers CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `-0.000451` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.002230` (lowers CT win probability)
- `lag_00__T_place_TSIDEUPPER`: coefficient `0.001524` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001480` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001277` (raises CT win probability)
- `lag_01__CT_place_UNKNOWN`: coefficient `-0.001070` (lowers CT win probability)
- `lag_01__T_place_SIDEENTRANCE`: coefficient `-0.001062` (lowers CT win probability)
- `lag_02__T_place_SIDEENTRANCE`: coefficient `-0.001048` (lowers CT win probability)
- `lag_02__CT_place_SIDEENTRANCE`: coefficient `0.000991` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000938` (lowers CT win probability)
- `lag_00__T_place_TUNNEL`: coefficient `-0.000930` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `125972`, seconds `78.50`, LSTM delta `-0.0944`

Top all feature movements:
- `lag_02__CT_place_SIDEENTRANCE`: contribution `-0.003987`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003146`
- `lag_00__T_kills_last_3s`: contribution `-0.002232`
- `lag_12__CT2__is_walking`: contribution `-0.002128`
- `lag_12__CT5__is_walking`: contribution `-0.002058`

Top utility-only movements:
- `lag_00__T_utility_damage_last_5s`: contribution `-0.003146`
- `lag_02__T_B_site_active_infernos`: contribution `-0.001443`

### tick `123988`, seconds `47.50`, LSTM delta `+0.0762`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.020388`
- `lag_00__T4__shots_fired`: contribution `+0.006647`
- `lag_00__T5__shots_fired`: contribution `+0.006467`
- `lag_08__T5__shots_fired`: contribution `+0.003459`
- `lag_00__T2__is_walking`: contribution `+0.002132`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.001729`
- `lag_13__T_active_infernos`: contribution `+0.000922`

### tick `124404`, seconds `54.00`, LSTM delta `-0.0750`

Top all feature movements:
- `lag_13__T_shots_fired_sum`: contribution `-0.014185`
- `lag_13__T5__shots_fired`: contribution `-0.004711`
- `lag_05__T_shots_fired_sum`: contribution `-0.003802`
- `lag_05__CT_place_SIDEHALL`: contribution `-0.003595`
- `lag_00__CT2__is_walking`: contribution `-0.003493`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120980`, seconds `0.50`, LSTM delta `-0.0580`

Top all feature movements:
- `lag_01__CT_place_UNKNOWN`: contribution `-0.037554`
- `lag_00__T_velocity_mean`: contribution `-0.000885`
- `lag_01__T_place_TSPAWN`: contribution `-0.000845`
- `lag_01__T_velocity_mean`: contribution `-0.000539`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000536`

Top utility-only movements:
- `lag_01__T1__smoke`: contribution `-0.000457`
- `lag_01__T3__molly`: contribution `+0.000258`
- `lag_01__active_smokes_total`: contribution `-0.000248`
- `lag_01__T1__molly`: contribution `+0.000236`

### tick `121972`, seconds `16.00`, LSTM delta `+0.0491`

Top all feature movements:
- `lag_00__T_place_TUNNEL`: contribution `+0.011293`
- `lag_07__T_place_TUNNEL`: contribution `+0.003683`
- `lag_00__CT2__is_walking`: contribution `+0.003493`
- `lag_00__T_place_WATER`: contribution `+0.002924`
- `lag_06__CT_place_SIDEHALL`: contribution `+0.002812`

Top utility-only movements:
- `lag_13__CT4__flash_duration`: contribution `+0.001261`
