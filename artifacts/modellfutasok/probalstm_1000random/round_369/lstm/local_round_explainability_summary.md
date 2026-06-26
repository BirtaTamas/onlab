# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-3dmax-vs-betboom-anubis-9yOMu3EhAmKzkIxUzvijXH/3dmax-vs-betboom-anubis.csv`
- round_num: `3`

## Largest probability jumps

- tick `15360`, seconds `14.00`, LSTM `0.2186`, delta `-0.1907`
- tick `16928`, seconds `38.50`, LSTM `0.0467`, delta `-0.1645`
- tick `15424`, seconds `15.00`, LSTM `0.1668`, delta `-0.0628`
- tick `16832`, seconds `37.00`, LSTM `0.1937`, delta `-0.0554`
- tick `14944`, seconds `7.50`, LSTM `0.3791`, delta `+0.0464`
- tick `14720`, seconds `4.00`, LSTM `0.3880`, delta `+0.0426`
- tick `16288`, seconds `28.50`, LSTM `0.2077`, delta `-0.0391`
- tick `15616`, seconds `18.00`, LSTM `0.1664`, delta `+0.0384`
- tick `16992`, seconds `39.50`, LSTM `0.0078`, delta `-0.0372`
- tick `16576`, seconds `33.00`, LSTM `0.2735`, delta `+0.0291`

## Top 15 local ridge features

- `lag_07__CT_place_MAIN`: coefficient `-0.001450`, |coef| `0.001450`
- `lag_10__CT_place_MAIN`: coefficient `-0.001422`, |coef| `0.001422`
- `lag_05__T_flashed_players`: coefficient `-0.001277`, |coef| `0.001277`
- `lag_08__T4__duck_amount`: coefficient `0.001248`, |coef| `0.001248`
- `lag_00__T_kills_last_3s`: coefficient `-0.001187`, |coef| `0.001187`
- `lag_00__CT4__flash_duration`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001125`, |coef| `0.001125`
- `lag_09__T4__duck_amount`: coefficient `-0.001120`, |coef| `0.001120`
- `lag_03__CT3__is_scoped`: coefficient `0.001105`, |coef| `0.001105`
- `lag_13__CT_place_BACKOFB`: coefficient `-0.001081`, |coef| `0.001081`
- `lag_03__T_place_CONNECTOR`: coefficient `-0.000993`, |coef| `0.000993`
- `lag_01__T_place_CONNECTOR`: coefficient `-0.000991`, |coef| `0.000991`
- `lag_12__CT_place_WALKWAY`: coefficient `0.000970`, |coef| `0.000970`
- `lag_00__kill_diff_last_3s`: coefficient `0.000938`, |coef| `0.000938`
- `lag_12__T_place_RUINS`: coefficient `0.000897`, |coef| `0.000897`

## Top 10 utility ridge features

- `lag_00__CT4__flash_duration`: coefficient `-0.001128` (lowers CT win probability)
- `lag_00__CT2__flash`: coefficient `0.000829` (raises CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000715` (lowers CT win probability)
- `lag_10__T_utility_damage_last_5s`: coefficient `0.000661` (raises CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `-0.000601` (lowers CT win probability)
- `lag_14__CT_utility_damage_last_5s`: coefficient `-0.000592` (lowers CT win probability)
- `lag_00__CT2__utility_total`: coefficient `0.000581` (raises CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.000581` (lowers CT win probability)
- `lag_04__CT1__molly`: coefficient `0.000570` (raises CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.000525` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_07__CT_place_MAIN`: coefficient `-0.001450` (lowers CT win probability)
- `lag_10__CT_place_MAIN`: coefficient `-0.001422` (lowers CT win probability)
- `lag_05__T_flashed_players`: coefficient `-0.001277` (lowers CT win probability)
- `lag_08__T4__duck_amount`: coefficient `0.001248` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001187` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001125` (raises CT win probability)
- `lag_09__T4__duck_amount`: coefficient `-0.001120` (lowers CT win probability)
- `lag_03__CT3__is_scoped`: coefficient `0.001105` (raises CT win probability)
- `lag_13__CT_place_BACKOFB`: coefficient `-0.001081` (lowers CT win probability)
- `lag_03__T_place_CONNECTOR`: coefficient `-0.000993` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15360`, seconds `14.00`, LSTM delta `-0.1907`

Top all feature movements:
- `lag_07__CT_place_MAIN`: contribution `-0.009762`
- `lag_00__CT4__flash_duration`: contribution `-0.007897`
- `lag_13__CT_place_BACKOFB`: contribution `-0.006170`
- `lag_12__CT_place_WALKWAY`: contribution `-0.004761`
- `lag_08__T_shots_fired_sum`: contribution `-0.004659`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `-0.007897`
- `lag_00__CT2__flash`: contribution `-0.002997`

### tick `16928`, seconds `38.50`, LSTM delta `-0.1645`

Top all feature movements:
- `lag_10__CT_place_MAIN`: contribution `-0.009576`
- `lag_05__T_flashed_players`: contribution `-0.007392`
- `lag_03__CT3__is_scoped`: contribution `-0.005027`
- `lag_03__T_place_CONNECTOR`: contribution `-0.004808`
- `lag_01__T_place_CONNECTOR`: contribution `-0.004801`

Top utility-only movements:
- `lag_10__T_utility_damage_last_5s`: contribution `-0.003682`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.002456`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.002046`

### tick `15424`, seconds `15.00`, LSTM delta `-0.0628`

Top all feature movements:
- `lag_03__T_shots_fired_sum`: contribution `-0.004390`
- `lag_02__T_shots_fired_sum`: contribution `-0.003999`
- `lag_09__CT_place_MAIN`: contribution `-0.003920`
- `lag_00__T_shots_fired_sum`: contribution `+0.003779`
- `lag_11__T_place_TSTAIRS`: contribution `-0.003268`

Top utility-only movements:
- `lag_02__CT4__flash_duration`: contribution `-0.002850`

### tick `16832`, seconds `37.00`, LSTM delta `-0.0554`

Top all feature movements:
- `lag_07__CT_place_MAIN`: contribution `-0.009762`
- `lag_00__CT3__is_scoped`: contribution `-0.002922`
- `lag_09__T1__duck_amount`: contribution `-0.002389`
- `lag_00__T_place_CONNECTOR`: contribution `-0.002196`
- `lag_03__T1__is_walking`: contribution `-0.001988`

Top utility-only movements:
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.001169`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.001167`
- `lag_07__T_utility_damage_last_5s`: contribution `-0.001072`

### tick `14944`, seconds `7.50`, LSTM delta `+0.0464`

Top all feature movements:
- `lag_15__CT_place_CTSIDEUPPER`: contribution `+0.005779`
- `lag_07__CT_place_LOWERTUNNEL`: contribution `+0.005590`
- `lag_11__CT_place_CTSIDEUPPER`: contribution `+0.004169`
- `lag_12__CT_place_LOWERTUNNEL`: contribution `+0.003628`
- `lag_00__T_place_STREET`: contribution `+0.003510`

Top utility-only movements:
- `lag_15__CT5__smoke`: contribution `+0.000669`
- `lag_15__CT2__smoke`: contribution `-0.000554`
- `lag_15__CT5__utility_total`: contribution `+0.000479`
