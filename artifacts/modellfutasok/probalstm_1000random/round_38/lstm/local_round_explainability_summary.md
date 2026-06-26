# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `4`

## Largest probability jumps

- tick `29471`, seconds `90.50`, LSTM `0.1741`, delta `-0.2785`
- tick `29087`, seconds `84.50`, LSTM `0.6975`, delta `+0.1730`
- tick `29151`, seconds `85.50`, LSTM `0.4999`, delta `-0.1491`
- tick `30207`, seconds `102.00`, LSTM `0.0409`, delta `-0.1344`
- tick `29055`, seconds `84.00`, LSTM `0.5245`, delta `+0.0596`
- tick `29279`, seconds `87.50`, LSTM `0.4636`, delta `+0.0592`
- tick `29791`, seconds `95.50`, LSTM `0.1741`, delta `+0.0571`
- tick `29023`, seconds `83.50`, LSTM `0.4649`, delta `-0.0499`
- tick `29119`, seconds `85.00`, LSTM `0.6490`, delta `-0.0486`
- tick `30143`, seconds `101.00`, LSTM `0.1500`, delta `-0.0409`

## Top 15 local ridge features

- `lag_03__CT_place_HOLE`: coefficient `0.002177`, |coef| `0.002177`
- `lag_00__T_kills_last_3s`: coefficient `-0.001711`, |coef| `0.001711`
- `lag_00__kill_diff_last_3s`: coefficient `0.001421`, |coef| `0.001421`
- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001304`, |coef| `0.001304`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001248`, |coef| `0.001248`
- `lag_13__CT_place_HOLE`: coefficient `-0.001198`, |coef| `0.001198`
- `lag_07__CT_place_HOLE`: coefficient `-0.001166`, |coef| `0.001166`
- `lag_02__T_place_ARAMP`: coefficient `-0.001157`, |coef| `0.001157`
- `lag_02__CT_place_BDOORS`: coefficient `-0.001148`, |coef| `0.001148`
- `lag_12__CT2__is_walking`: coefficient `0.001123`, |coef| `0.001123`
- `lag_05__T3__duck_amount`: coefficient `0.001122`, |coef| `0.001122`
- `lag_00__damage_diff_last_5s`: coefficient `0.001109`, |coef| `0.001109`
- `lag_11__T_place_LONGA`: coefficient `-0.001107`, |coef| `0.001107`
- `lag_08__CT_place_BDOORS`: coefficient `-0.001106`, |coef| `0.001106`
- `lag_14__CT_flashed_players`: coefficient `0.001103`, |coef| `0.001103`

## Top 10 utility ridge features

- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001304` (raises CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.001059` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.000986` (lowers CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.000867` (raises CT win probability)
- `lag_04__utility_damage_diff_last_5s`: coefficient `-0.000822` (lowers CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.000791` (lowers CT win probability)
- `lag_06__CT_utility_damage_last_5s`: coefficient `0.000723` (raises CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.000717` (lowers CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `0.000706` (raises CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `0.000700` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__CT_place_HOLE`: coefficient `0.002177` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001711` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001421` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001248` (raises CT win probability)
- `lag_13__CT_place_HOLE`: coefficient `-0.001198` (lowers CT win probability)
- `lag_07__CT_place_HOLE`: coefficient `-0.001166` (lowers CT win probability)
- `lag_02__T_place_ARAMP`: coefficient `-0.001157` (lowers CT win probability)
- `lag_02__CT_place_BDOORS`: coefficient `-0.001148` (lowers CT win probability)
- `lag_12__CT2__is_walking`: coefficient `0.001123` (raises CT win probability)
- `lag_05__T3__duck_amount`: coefficient `0.001122` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `29471`, seconds `90.50`, LSTM delta `-0.2785`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `-0.024306`
- `lag_07__CT_place_HOLE`: contribution `-0.013013`
- `lag_14__CT_flashed_players`: contribution `-0.007243`
- `lag_00__T_kills_last_3s`: contribution `-0.005421`
- `lag_00__T_place_EXTENDEDA`: contribution `-0.005210`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `29087`, seconds `84.50`, LSTM delta `+0.1730`

Top all feature movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.008609`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.006510`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.005736`
- `lag_02__CT_place_BDOORS`: contribution `+0.005524`
- `lag_10__CT_place_ARAMP`: contribution `+0.004585`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.008609`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.006510`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.005736`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.004455`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.003533`

### tick `29151`, seconds `85.50`, LSTM delta `-0.1491`

Top all feature movements:
- `lag_12__CT_place_ARAMP`: contribution `-0.006462`
- `lag_15__CT_place_ARAMP`: contribution `-0.005845`
- `lag_00__T_kills_last_3s`: contribution `-0.005421`
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.004774`
- `lag_04__CT_place_BDOORS`: contribution `-0.003912`

Top utility-only movements:
- `lag_06__CT_utility_damage_last_5s`: contribution `-0.004774`
- `lag_06__utility_damage_diff_last_5s`: contribution `-0.003152`
- `lag_07__CT_utility_damage_last_5s`: contribution `-0.001917`
- `lag_10__T_A_site_active_infernos`: contribution `-0.001860`

### tick `30207`, seconds `102.00`, LSTM delta `-0.1344`

Top all feature movements:
- `lag_02__T_place_ARAMP`: contribution `-0.010469`
- `lag_08__T1__is_scoped`: contribution `-0.006117`
- `lag_00__T_kills_last_3s`: contribution `-0.005421`
- `lag_11__T_place_LONGA`: contribution `-0.004715`
- `lag_00__T_A_site_active_infernos`: contribution `-0.004271`

Top utility-only movements:
- `lag_00__T_A_site_active_infernos`: contribution `-0.004271`
- `lag_00__T_active_infernos`: contribution `-0.002091`

### tick `29055`, seconds `84.00`, LSTM delta `+0.0596`

Top all feature movements:
- `lag_12__CT_place_ARAMP`: contribution `+0.006462`
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.005309`
- `lag_01__CT_flashed_players`: contribution `+0.004120`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.004015`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.003537`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.005309`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.004015`
- `lag_14__utility_damage_diff_last_5s`: contribution `+0.003537`
- `lag_13__CT_utility_damage_last_5s`: contribution `+0.003452`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.002747`
