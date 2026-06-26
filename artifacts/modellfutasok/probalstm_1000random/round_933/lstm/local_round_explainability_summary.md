# Local Round Explainability

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `21`

## Largest probability jumps

- tick `166214`, seconds `15.50`, LSTM `0.1730`, delta `-0.3347`
- tick `166246`, seconds `16.00`, LSTM `0.0301`, delta `-0.1429`
- tick `167494`, seconds `35.50`, LSTM `0.0863`, delta `-0.0468`
- tick `166182`, seconds `15.00`, LSTM `0.5077`, delta `-0.0409`
- tick `167110`, seconds `29.50`, LSTM `0.1527`, delta `+0.0361`
- tick `165926`, seconds `11.00`, LSTM `0.5530`, delta `+0.0321`
- tick `167750`, seconds `39.50`, LSTM `0.0358`, delta `-0.0308`
- tick `166886`, seconds `26.00`, LSTM `0.0967`, delta `+0.0285`
- tick `166726`, seconds `23.50`, LSTM `0.0377`, delta `+0.0234`
- tick `167462`, seconds `35.00`, LSTM `0.1331`, delta `-0.0234`

## Top 15 local ridge features

- `lag_12__CT_place_ADMIN`: coefficient `0.001232`, |coef| `0.001232`
- `lag_09__CT_A_site_active_infernos`: coefficient `-0.001125`, |coef| `0.001125`
- `lag_00__T_kills_last_3s`: coefficient `-0.001083`, |coef| `0.001083`
- `lag_09__T4__flash_duration`: coefficient `-0.001040`, |coef| `0.001040`
- `lag_05__CT_place_SQUEAKY`: coefficient `-0.001027`, |coef| `0.001027`
- `lag_09__T_burning_players`: coefficient `-0.001018`, |coef| `0.001018`
- `lag_07__T_place_GARAGE`: coefficient `-0.001015`, |coef| `0.001015`
- `lag_05__T_place_SILO`: coefficient `-0.001014`, |coef| `0.001014`
- `lag_15__CT_place_RAFTERS`: coefficient `-0.001009`, |coef| `0.001009`
- `lag_09__T1__flash_duration`: coefficient `-0.001004`, |coef| `0.001004`
- `lag_08__T_place_SQUEAKY`: coefficient `0.001003`, |coef| `0.001003`
- `lag_15__CT_place_SQUEAKY`: coefficient `-0.001002`, |coef| `0.001002`
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.000997`, |coef| `0.000997`
- `lag_13__CT_place_SQUEAKY`: coefficient `-0.000990`, |coef| `0.000990`
- `lag_11__CT1__flash_duration`: coefficient `-0.000961`, |coef| `0.000961`

## Top 10 utility ridge features

- `lag_09__CT_A_site_active_infernos`: coefficient `-0.001125` (lowers CT win probability)
- `lag_09__T4__flash_duration`: coefficient `-0.001040` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `-0.001004` (lowers CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.000997` (lowers CT win probability)
- `lag_11__CT1__flash_duration`: coefficient `-0.000961` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `-0.000883` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `-0.000858` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000815` (raises CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `-0.000813` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.000793` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_ADMIN`: coefficient `0.001232` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001083` (lowers CT win probability)
- `lag_05__CT_place_SQUEAKY`: coefficient `-0.001027` (lowers CT win probability)
- `lag_09__T_burning_players`: coefficient `-0.001018` (lowers CT win probability)
- `lag_07__T_place_GARAGE`: coefficient `-0.001015` (lowers CT win probability)
- `lag_05__T_place_SILO`: coefficient `-0.001014` (lowers CT win probability)
- `lag_15__CT_place_RAFTERS`: coefficient `-0.001009` (lowers CT win probability)
- `lag_08__T_place_SQUEAKY`: coefficient `0.001003` (raises CT win probability)
- `lag_15__CT_place_SQUEAKY`: coefficient `-0.001002` (lowers CT win probability)
- `lag_13__CT_place_SQUEAKY`: coefficient `-0.000990` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `166214`, seconds `15.50`, LSTM delta `-0.3347`

Top all feature movements:
- `lag_12__CT_place_ADMIN`: contribution `-0.008556`
- `lag_09__CT_A_site_active_infernos`: contribution `-0.007939`
- `lag_09__T4__flash_duration`: contribution `-0.007283`
- `lag_05__T_place_SILO`: contribution `-0.006891`
- `lag_00__T_kills_last_3s`: contribution `-0.006860`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.007939`
- `lag_09__T4__flash_duration`: contribution `-0.007283`
- `lag_09__CT_B_site_active_infernos`: contribution `-0.006850`
- `lag_09__T1__flash_duration`: contribution `-0.006492`
- `lag_11__CT1__flash_duration`: contribution `-0.006270`

### tick `166246`, seconds `16.00`, LSTM delta `-0.1429`

Top all feature movements:
- `lag_08__T_place_SQUEAKY`: contribution `-0.006244`
- `lag_15__CT_place_RAFTERS`: contribution `-0.005392`
- `lag_01__T_shots_fired_sum`: contribution `+0.004748`
- `lag_15__CT_place_HEAVEN`: contribution `-0.004261`
- `lag_13__CT_place_ADMIN`: contribution `-0.004228`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `-0.003862`
- `lag_10__CT_A_site_active_infernos`: contribution `-0.003712`
- `lag_10__T4__flash_duration`: contribution `-0.003543`
- `lag_10__CT_B_site_active_infernos`: contribution `-0.002813`
- `lag_12__CT1__flash_duration`: contribution `-0.002691`

### tick `167494`, seconds `35.50`, LSTM delta `-0.0468`

Top all feature movements:
- `lag_05__CT_place_SQUEAKY`: contribution `-0.013660`
- `lag_12__CT_place_ADMIN`: contribution `-0.008556`
- `lag_01__CT_place_CONTROL`: contribution `-0.008188`
- `lag_12__CT_place_RAMP`: contribution `-0.001813`
- `lag_12__T_place_MINI`: contribution `-0.000896`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.000491`
- `lag_15__T4__molly`: contribution `-0.000437`
- `lag_05__T_B_site_active_infernos`: contribution `-0.000431`

### tick `166182`, seconds `15.00`, LSTM delta `-0.0409`

Top all feature movements:
- `lag_08__T_place_SQUEAKY`: contribution `+0.006244`
- `lag_07__T_place_SQUEAKY`: contribution `-0.005821`
- `lag_00__T_shots_fired_sum`: contribution `+0.005536`
- `lag_14__CT_place_RAFTERS`: contribution `-0.004614`
- `lag_14__CT_place_HEAVEN`: contribution `-0.002801`

Top utility-only movements:
- `lag_15__CT_A_site_active_infernos`: contribution `-0.002799`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.002081`
- `lag_08__CT_utility_damage_last_5s`: contribution `-0.001402`
- `lag_08__T4__flash_duration`: contribution `-0.001385`
- `lag_10__CT1__flash_duration`: contribution `-0.001384`

### tick `167110`, seconds `29.50`, LSTM delta `+0.0361`

Top all feature movements:
- `lag_10__CT_place_TROPHY`: contribution `+0.004680`
- `lag_00__T_place_MINI`: contribution `+0.004372`
- `lag_07__CT_place_ADMIN`: contribution `+0.004347`
- `lag_04__CT_place_VENDING`: contribution `+0.004330`
- `lag_10__CT_place_VENDING`: contribution `-0.003160`

Top utility-only movements:
- `lag_00__T_A_site_active_smokes`: contribution `-0.000457`
