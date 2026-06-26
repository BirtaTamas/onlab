# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `4`

## Largest probability jumps

- tick `17481`, seconds `23.00`, LSTM `0.0959`, delta `-0.2128`
- tick `17161`, seconds `18.00`, LSTM `0.3928`, delta `-0.1284`
- tick `18217`, seconds `34.50`, LSTM `0.1519`, delta `+0.0865`
- tick `17449`, seconds `22.50`, LSTM `0.3087`, delta `-0.0818`
- tick `19849`, seconds `60.00`, LSTM `0.0499`, delta `-0.0622`
- tick `19817`, seconds `59.50`, LSTM `0.1121`, delta `-0.0516`
- tick `18313`, seconds `36.00`, LSTM `0.2059`, delta `+0.0449`
- tick `17353`, seconds `21.00`, LSTM `0.3867`, delta `+0.0421`
- tick `19561`, seconds `55.50`, LSTM `0.1819`, delta `+0.0370`
- tick `17225`, seconds `19.00`, LSTM `0.3859`, delta `-0.0359`

## Top 15 local ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003073`, |coef| `0.003073`
- `lag_10__CT_place_MAINHALL`: coefficient `0.002291`, |coef| `0.002291`
- `lag_09__CT2__flash_duration`: coefficient `0.001873`, |coef| `0.001873`
- `lag_00__kill_diff_last_3s`: coefficient `0.001841`, |coef| `0.001841`
- `lag_00__T_kills_last_3s`: coefficient `-0.001721`, |coef| `0.001721`
- `lag_11__CT3__flash_duration`: coefficient `0.001716`, |coef| `0.001716`
- `lag_01__CT5__shots_fired`: coefficient `-0.001443`, |coef| `0.001443`
- `lag_04__CT4__flash_duration`: coefficient `-0.001331`, |coef| `0.001331`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001151`, |coef| `0.001151`
- `lag_00__CT_place_MAINHALL`: coefficient `0.001145`, |coef| `0.001145`
- `lag_14__CT3__is_walking`: coefficient `0.001128`, |coef| `0.001128`
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001106`, |coef| `0.001106`
- `lag_04__CT_place_SIDEHALL`: coefficient `-0.001093`, |coef| `0.001093`
- `lag_09__T_shots_fired_sum`: coefficient `0.001075`, |coef| `0.001075`
- `lag_10__CT1__duck_amount`: coefficient `0.001048`, |coef| `0.001048`

## Top 10 utility ridge features

- `lag_09__CT2__flash_duration`: coefficient `0.001873` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `0.001716` (raises CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `-0.001331` (lowers CT win probability)
- `lag_10__CT3__flash_duration`: coefficient `0.001000` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000922` (raises CT win probability)
- `lag_10__CT1__utility_total`: coefficient `0.000922` (raises CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `-0.000915` (lowers CT win probability)
- `lag_08__CT2__flash_duration`: coefficient `0.000863` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000816` (raises CT win probability)
- `lag_09__CT_flash_duration_sum`: coefficient `0.000804` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_UNKNOWN`: coefficient `0.003073` (raises CT win probability)
- `lag_10__CT_place_MAINHALL`: coefficient `0.002291` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001841` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001721` (lowers CT win probability)
- `lag_01__CT5__shots_fired`: coefficient `-0.001443` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001151` (lowers CT win probability)
- `lag_00__CT_place_MAINHALL`: coefficient `0.001145` (raises CT win probability)
- `lag_14__CT3__is_walking`: coefficient `0.001128` (raises CT win probability)
- `lag_00__CT_place_SIDEENTRANCE`: coefficient `0.001106` (raises CT win probability)
- `lag_04__CT_place_SIDEHALL`: coefficient `-0.001093` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `17481`, seconds `23.00`, LSTM delta `-0.2128`

Top all feature movements:
- `lag_10__CT_place_MAINHALL`: contribution `-0.018965`
- `lag_09__CT2__flash_duration`: contribution `-0.013722`
- `lag_11__CT3__flash_duration`: contribution `-0.009854`
- `lag_09__T_shots_fired_sum`: contribution `-0.006448`
- `lag_00__T_kills_last_3s`: contribution `-0.005454`

Top utility-only movements:
- `lag_09__CT2__flash_duration`: contribution `-0.013722`
- `lag_11__CT3__flash_duration`: contribution `-0.009854`
- `lag_09__CT_flash_duration_sum`: contribution `-0.002674`
- `lag_10__CT1__utility_total`: contribution `-0.002596`

### tick `17161`, seconds `18.00`, LSTM delta `-0.1284`

Top all feature movements:
- `lag_00__CT_place_MAINHALL`: contribution `-0.009477`
- `lag_12__CT2__flash_duration`: contribution `-0.006706`
- `lag_00__T_kills_last_3s`: contribution `-0.005454`
- `lag_00__kill_diff_last_3s`: contribution `-0.004430`
- `lag_01__CT3__flash_duration`: contribution `-0.004393`

Top utility-only movements:
- `lag_12__CT2__flash_duration`: contribution `-0.006706`
- `lag_01__CT3__flash_duration`: contribution `-0.004393`
- `lag_09__CT_active_infernos`: contribution `-0.002570`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.002545`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.002443`

### tick `18217`, seconds `34.50`, LSTM delta `+0.0865`

Top all feature movements:
- `lag_04__CT4__flash_duration`: contribution `+0.009200`
- `lag_04__CT_place_SIDEHALL`: contribution `+0.004674`
- `lag_00__kill_diff_last_3s`: contribution `+0.004430`
- `lag_00__T_place_HOUSE`: contribution `+0.003891`
- `lag_12__T_place_MAINHALL`: contribution `+0.003640`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `+0.009200`
- `lag_04__CT_flash_duration_sum`: contribution `+0.002089`
- `lag_13__T_A_site_active_infernos`: contribution `+0.002086`
- `lag_07__CT_A_site_active_infernos`: contribution `+0.001711`
- `lag_10__CT3__molly`: contribution `+0.001442`

### tick `17449`, seconds `22.50`, LSTM delta `-0.0818`

Top all feature movements:
- `lag_09__CT_place_MAINHALL`: contribution `-0.007973`
- `lag_08__CT2__flash_duration`: contribution `-0.006321`
- `lag_10__CT3__flash_duration`: contribution `-0.005745`
- `lag_09__T_shots_fired_sum`: contribution `+0.004030`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003607`

Top utility-only movements:
- `lag_08__CT2__flash_duration`: contribution `-0.006321`
- `lag_10__CT3__flash_duration`: contribution `-0.005745`
- `lag_08__CT_flash_duration_sum`: contribution `-0.001478`

### tick `19849`, seconds `60.00`, LSTM delta `-0.0622`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `-0.007001`
- `lag_00__T_kills_last_3s`: contribution `-0.005454`
- `lag_01__T_shots_fired_sum`: contribution `-0.005177`
- `lag_00__kill_diff_last_3s`: contribution `-0.004430`
- `lag_00__T4__flash_duration`: contribution `-0.003795`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.003795`
