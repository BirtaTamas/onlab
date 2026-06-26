# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `4`

## Largest probability jumps

- tick `16701`, seconds `28.50`, LSTM `0.8864`, delta `+0.1254`
- tick `18141`, seconds `51.00`, LSTM `0.9438`, delta `+0.1077`
- tick `21853`, seconds `109.00`, LSTM `0.8908`, delta `+0.1061`
- tick `15901`, seconds `16.00`, LSTM `0.6446`, delta `+0.0946`
- tick `21341`, seconds `101.00`, LSTM `0.8422`, delta `-0.0928`
- tick `17725`, seconds `44.50`, LSTM `0.8278`, delta `-0.0922`
- tick `16509`, seconds `25.50`, LSTM `0.7757`, delta `+0.0662`
- tick `16477`, seconds `25.00`, LSTM `0.7095`, delta `+0.0425`
- tick `16253`, seconds `21.50`, LSTM `0.7249`, delta `+0.0410`
- tick `21373`, seconds `101.50`, LSTM `0.8056`, delta `-0.0366`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001893`, |coef| `0.001893`
- `lag_03__CT_place_SIDEALLEY`: coefficient `-0.001687`, |coef| `0.001687`
- `lag_03__CT_place_PALACEALLEY`: coefficient `0.001400`, |coef| `0.001400`
- `lag_00__CT_place_JUNGLE`: coefficient `0.001293`, |coef| `0.001293`
- `lag_06__CT_place_SCAFFOLDING`: coefficient `0.001290`, |coef| `0.001290`
- `lag_00__CT_kills_last_3s`: coefficient `0.001212`, |coef| `0.001212`
- `lag_03__CT_place_TSPAWN`: coefficient `-0.001211`, |coef| `0.001211`
- `lag_11__CT_place_SIDEALLEY`: coefficient `-0.001203`, |coef| `0.001203`
- `lag_02__CT_place_SCAFFOLDING`: coefficient `-0.001202`, |coef| `0.001202`
- `lag_00__T_kills_last_3s`: coefficient `-0.001162`, |coef| `0.001162`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001078`, |coef| `0.001078`
- `lag_00__damage_diff_last_5s`: coefficient `0.001076`, |coef| `0.001076`
- `lag_00__CT_place_SIDEALLEY`: coefficient `0.001056`, |coef| `0.001056`
- `lag_15__CT_place_SIDEALLEY`: coefficient `-0.001018`, |coef| `0.001018`
- `lag_00__CT3__duck_amount`: coefficient `0.001005`, |coef| `0.001005`

## Top 10 utility ridge features

- `lag_03__CT_active_infernos`: coefficient `-0.000614` (lowers CT win probability)
- `lag_07__CT_utility_damage_last_5s`: coefficient `0.000580` (raises CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `-0.000547` (lowers CT win probability)
- `lag_06__T5__flash_duration`: coefficient `0.000513` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.000511` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000501` (lowers CT win probability)
- `lag_07__utility_damage_diff_last_5s`: coefficient `0.000486` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.000466` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000460` (lowers CT win probability)
- `lag_14__T1__molly`: coefficient `0.000457` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001893` (raises CT win probability)
- `lag_03__CT_place_SIDEALLEY`: coefficient `-0.001687` (lowers CT win probability)
- `lag_03__CT_place_PALACEALLEY`: coefficient `0.001400` (raises CT win probability)
- `lag_00__CT_place_JUNGLE`: coefficient `0.001293` (raises CT win probability)
- `lag_06__CT_place_SCAFFOLDING`: coefficient `0.001290` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001212` (raises CT win probability)
- `lag_03__CT_place_TSPAWN`: coefficient `-0.001211` (lowers CT win probability)
- `lag_11__CT_place_SIDEALLEY`: coefficient `-0.001203` (lowers CT win probability)
- `lag_02__CT_place_SCAFFOLDING`: coefficient `-0.001202` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001162` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `16701`, seconds `28.50`, LSTM delta `+0.1254`

Top all feature movements:
- `lag_06__T_shots_fired_sum`: contribution `+0.010636`
- `lag_06__T5__shots_fired`: contribution `+0.006752`
- `lag_00__kill_diff_last_3s`: contribution `+0.004557`
- `lag_00__CT_kills_last_3s`: contribution `+0.003499`
- `lag_14__CT_place_TRUCK`: contribution `+0.003295`

Top utility-only movements:
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.002927`
- `lag_07__CT_utility_damage_last_5s`: contribution `+0.002363`
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.002226`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.001924`
- `lag_07__utility_damage_diff_last_5s`: contribution `+0.001625`

### tick `18141`, seconds `51.00`, LSTM delta `+0.1077`

Top all feature movements:
- `lag_06__CT_place_SCAFFOLDING`: contribution `+0.026928`
- `lag_02__CT_place_SCAFFOLDING`: contribution `+0.025075`
- `lag_06__T_place_JUNGLE`: contribution `+0.007359`
- `lag_00__kill_diff_last_3s`: contribution `+0.004557`
- `lag_00__CT_kills_last_3s`: contribution `+0.003499`

Top utility-only movements:
- `lag_05__CT_A_site_active_infernos`: contribution `+0.001229`
- `lag_05__CT_active_infernos`: contribution `+0.000969`

### tick `21853`, seconds `109.00`, LSTM delta `+0.1061`

Top all feature movements:
- `lag_03__CT_place_PALACEALLEY`: contribution `+0.021372`
- `lag_15__CT_place_SIDEALLEY`: contribution `+0.018565`
- `lag_03__CT_place_TSPAWN`: contribution `+0.009065`
- `lag_15__CT_place_TSPAWN`: contribution `+0.005107`
- `lag_00__kill_diff_last_3s`: contribution `+0.004557`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.001132`

### tick `15901`, seconds `16.00`, LSTM delta `+0.0946`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.004557`
- `lag_00__CT_kills_last_3s`: contribution `+0.003499`
- `lag_00__CT_place_SHOP`: contribution `+0.002949`
- `lag_06__T5__flash_duration`: contribution `+0.002890`
- `lag_13__T_place_SIDEALLEY`: contribution `+0.002686`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.002890`
- `lag_00__T3__flash_duration`: contribution `+0.002457`
- `lag_04__T3__flash_duration`: contribution `+0.001779`
- `lag_09__T4__flash_duration`: contribution `+0.001748`
- `lag_10__T3__flash_duration`: contribution `+0.001315`

### tick `21341`, seconds `101.00`, LSTM delta `-0.0928`

Top all feature movements:
- `lag_03__CT_place_SIDEALLEY`: contribution `-0.030783`
- `lag_02__T_bomb_zone_count`: contribution `-0.005267`
- `lag_00__kill_diff_last_3s`: contribution `-0.004557`
- `lag_10__T_place_CONNECTOR`: contribution `-0.003899`
- `lag_00__T_kills_last_3s`: contribution `-0.003682`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `-0.001318`
