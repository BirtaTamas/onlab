# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-g2-vs-betboom-bo3-pCfbtiY01aL_JW2Hy1pnZ6/g2-vs-betboom-m1-anubis.csv`
- round_num: `8`

## Largest probability jumps

- tick `73731`, seconds `45.50`, LSTM `0.6421`, delta `+0.4169`
- tick `74179`, seconds `52.50`, LSTM `0.8160`, delta `+0.2065`
- tick `74211`, seconds `53.00`, LSTM `0.6152`, delta `-0.2008`
- tick `76963`, seconds `96.00`, LSTM `0.3755`, delta `-0.1653`
- tick `77443`, seconds `103.50`, LSTM `0.0268`, delta `-0.1376`
- tick `73411`, seconds `40.50`, LSTM `0.1493`, delta `+0.0729`
- tick `77251`, seconds `100.50`, LSTM `0.2365`, delta `+0.0670`
- tick `70851`, seconds `0.50`, LSTM `0.1667`, delta `-0.0620`
- tick `73763`, seconds `46.00`, LSTM `0.5829`, delta `-0.0591`
- tick `73475`, seconds `41.50`, LSTM `0.2165`, delta `+0.0589`

## Top 15 local ridge features

- `lag_09__CT_place_BRICKS`: coefficient `0.004254`, |coef| `0.004254`
- `lag_07__CT_place_BRICKS`: coefficient `-0.004238`, |coef| `0.004238`
- `lag_00__CT_place_HEAVEN`: coefficient `0.003828`, |coef| `0.003828`
- `lag_00__kill_diff_last_3s`: coefficient `0.003815`, |coef| `0.003815`
- `lag_00__T_kills_last_3s`: coefficient `-0.002693`, |coef| `0.002693`
- `lag_03__CT2__duck_amount`: coefficient `0.002445`, |coef| `0.002445`
- `lag_04__T_place_FOUNTAIN`: coefficient `-0.002358`, |coef| `0.002358`
- `lag_00__CT_burning_players`: coefficient `0.002185`, |coef| `0.002185`
- `lag_00__CT_kills_last_3s`: coefficient `0.002122`, |coef| `0.002122`
- `lag_06__CT4__is_walking`: coefficient `0.002077`, |coef| `0.002077`
- `lag_08__CT_place_CANAL`: coefficient `-0.001995`, |coef| `0.001995`
- `lag_00__CT_place_TUNNEL`: coefficient `0.001951`, |coef| `0.001951`
- `lag_14__CT1__duck_amount`: coefficient `0.001943`, |coef| `0.001943`
- `lag_00__CT_place_MAIN`: coefficient `-0.001937`, |coef| `0.001937`
- `lag_01__CT_place_BRICKS`: coefficient `0.001901`, |coef| `0.001901`

## Top 10 utility ridge features

- `lag_03__T_active_infernos`: coefficient `-0.001495` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001332` (lowers CT win probability)
- `lag_05__T4__molly`: coefficient `0.001218` (raises CT win probability)
- `lag_13__T_utility_damage_last_5s`: coefficient `-0.001120` (lowers CT win probability)
- `lag_15__CT_B_site_active_smokes`: coefficient `0.001118` (raises CT win probability)
- `lag_11__T_active_infernos`: coefficient `-0.001113` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.001099` (raises CT win probability)
- `lag_03__active_infernos_total`: coefficient `-0.001034` (lowers CT win probability)
- `lag_15__T_utility_damage_last_5s`: coefficient `-0.001028` (lowers CT win probability)
- `lag_14__CT4__flash_duration`: coefficient `-0.000987` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__CT_place_BRICKS`: coefficient `0.004254` (raises CT win probability)
- `lag_07__CT_place_BRICKS`: coefficient `-0.004238` (lowers CT win probability)
- `lag_00__CT_place_HEAVEN`: coefficient `0.003828` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003815` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002693` (lowers CT win probability)
- `lag_03__CT2__duck_amount`: coefficient `0.002445` (raises CT win probability)
- `lag_04__T_place_FOUNTAIN`: coefficient `-0.002358` (lowers CT win probability)
- `lag_00__CT_burning_players`: coefficient `0.002185` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002122` (raises CT win probability)
- `lag_06__CT4__is_walking`: coefficient `0.002077` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `73731`, seconds `45.50`, LSTM delta `+0.4169`

Top all feature movements:
- `lag_09__CT_place_BRICKS`: contribution `+0.081681`
- `lag_07__CT_place_BRICKS`: contribution `+0.081377`
- `lag_00__CT_place_MAIN`: contribution `+0.013041`
- `lag_00__kill_diff_last_3s`: contribution `+0.009184`
- `lag_06__T_place_TSTAIRS`: contribution `+0.008580`

Top utility-only movements:
- `lag_11__T_active_infernos`: contribution `+0.004637`
- `lag_14__CT4__flash_duration`: contribution `+0.004558`

### tick `74179`, seconds `52.50`, LSTM delta `+0.2065`

Top all feature movements:
- `lag_14__CT_place_MAIN`: contribution `+0.011955`
- `lag_00__kill_diff_last_3s`: contribution `+0.009184`
- `lag_14__CT1__duck_amount`: contribution `+0.007412`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007384`
- `lag_00__T_place_BRIDGE`: contribution `+0.006561`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `74211`, seconds `53.00`, LSTM delta `-0.2008`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009184`
- `lag_01__T_shots_fired_sum`: contribution `-0.008907`
- `lag_00__T_kills_last_3s`: contribution `-0.008532`
- `lag_14__CT1__duck_amount`: contribution `-0.007412`
- `lag_01__CT_shots_fired_sum`: contribution `-0.007384`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `76963`, seconds `96.00`, LSTM delta `-0.1653`

Top all feature movements:
- `lag_00__CT_place_HEAVEN`: contribution `-0.020671`
- `lag_02__T_place_MAIN`: contribution `-0.011976`
- `lag_04__T_place_FOUNTAIN`: contribution `-0.011146`
- `lag_03__CT2__duck_amount`: contribution `-0.009314`
- `lag_00__kill_diff_last_3s`: contribution `-0.009184`

Top utility-only movements:
- `lag_03__T_A_site_active_infernos`: contribution `-0.003965`
- `lag_03__T_active_infernos`: contribution `-0.003113`
- `lag_05__T4__molly`: contribution `-0.002655`

### tick `77443`, seconds `103.50`, LSTM delta `-0.1376`

Top all feature movements:
- `lag_00__CT_place_HEAVEN`: contribution `-0.020671`
- `lag_08__CT_place_CANAL`: contribution `-0.012126`
- `lag_13__T_place_MAIN`: contribution `-0.011053`
- `lag_10__T_bomb_zone_count`: contribution `-0.009274`
- `lag_00__kill_diff_last_3s`: contribution `-0.009184`

Top utility-only movements:
- No utility movement among the top local contributors.
