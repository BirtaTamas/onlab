# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `16`

## Largest probability jumps

- tick `130539`, seconds `15.00`, LSTM `0.2624`, delta `-0.2021`
- tick `130571`, seconds `15.50`, LSTM `0.1958`, delta `-0.0665`
- tick `132907`, seconds `52.00`, LSTM `0.2318`, delta `+0.0632`
- tick `131979`, seconds `37.50`, LSTM `0.2572`, delta `-0.0629`
- tick `131947`, seconds `37.00`, LSTM `0.3201`, delta `+0.0563`
- tick `131083`, seconds `23.50`, LSTM `0.2202`, delta `+0.0485`
- tick `133035`, seconds `54.00`, LSTM `0.1432`, delta `-0.0425`
- tick `133067`, seconds `54.50`, LSTM `0.1054`, delta `-0.0378`
- tick `131915`, seconds `36.50`, LSTM `0.2638`, delta `+0.0377`
- tick `131179`, seconds `25.00`, LSTM `0.2452`, delta `+0.0334`

## Top 15 local ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001578`, |coef| `0.001578`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001453`, |coef| `0.001453`
- `lag_08__CT_place_MAIN`: coefficient `-0.001315`, |coef| `0.001315`
- `lag_00__CT3__flash`: coefficient `0.001072`, |coef| `0.001072`
- `lag_04__CT2__duck_amount`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_15__T_place_TSTAIRS`: coefficient `-0.000996`, |coef| `0.000996`
- `lag_08__CT_place_HEAVEN`: coefficient `-0.000949`, |coef| `0.000949`
- `lag_00__CT3__utility_total`: coefficient `0.000927`, |coef| `0.000927`
- `lag_15__T_place_STREET`: coefficient `0.000918`, |coef| `0.000918`
- `lag_13__T_place_CANAL`: coefficient `-0.000908`, |coef| `0.000908`
- `lag_05__T_place_MAIN`: coefficient `-0.000897`, |coef| `0.000897`
- `lag_04__CT_place_WALKWAY`: coefficient `-0.000887`, |coef| `0.000887`
- `lag_08__T3__duck_amount`: coefficient `-0.000854`, |coef| `0.000854`
- `lag_06__CT_place_BRIDGE`: coefficient `0.000822`, |coef| `0.000822`
- `lag_07__CT_place_BACKOFB`: coefficient `0.000814`, |coef| `0.000814`

## Top 10 utility ridge features

- `lag_00__CT3__flash`: coefficient `0.001072` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000927` (raises CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `-0.000804` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000742` (lowers CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000650` (lowers CT win probability)
- `lag_09__utility_damage_diff_last_5s`: coefficient `-0.000647` (lowers CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000643` (raises CT win probability)
- `lag_02__T_B_site_active_infernos`: coefficient `0.000619` (raises CT win probability)
- `lag_06__CT3__molly`: coefficient `0.000575` (raises CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.000553` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001578` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001453` (raises CT win probability)
- `lag_08__CT_place_MAIN`: coefficient `-0.001315` (lowers CT win probability)
- `lag_04__CT2__duck_amount`: coefficient `-0.001036` (lowers CT win probability)
- `lag_15__T_place_TSTAIRS`: coefficient `-0.000996` (lowers CT win probability)
- `lag_08__CT_place_HEAVEN`: coefficient `-0.000949` (lowers CT win probability)
- `lag_15__T_place_STREET`: coefficient `0.000918` (raises CT win probability)
- `lag_13__T_place_CANAL`: coefficient `-0.000908` (lowers CT win probability)
- `lag_05__T_place_MAIN`: coefficient `-0.000897` (lowers CT win probability)
- `lag_04__CT_place_WALKWAY`: coefficient `-0.000887` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `130539`, seconds `15.00`, LSTM delta `-0.2021`

Top all feature movements:
- `lag_08__CT_place_MAIN`: contribution `-0.008855`
- `lag_15__T_place_TSTAIRS`: contribution `-0.005648`
- `lag_08__CT_place_HEAVEN`: contribution `-0.005122`
- `lag_08__T3__is_scoped`: contribution `-0.005070`
- `lag_15__T_place_STREET`: contribution `-0.005047`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.003958`
- `lag_09__CT_utility_damage_last_5s`: contribution `-0.003452`
- `lag_00__CT3__utility_total`: contribution `-0.002655`
- `lag_02__CT_A_site_active_infernos`: contribution `-0.002617`

### tick `130571`, seconds `15.50`, LSTM delta `-0.0665`

Top all feature movements:
- `lag_09__CT_place_MAIN`: contribution `-0.004259`
- `lag_09__CT_place_HEAVEN`: contribution `-0.004156`
- `lag_13__T_place_CANAL`: contribution `-0.002526`
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.002293`
- `lag_05__CT_place_WALKWAY`: contribution `-0.002234`

Top utility-only movements:
- `lag_10__CT_utility_damage_last_5s`: contribution `-0.002293`
- `lag_01__CT3__flash`: contribution `-0.001679`
- `lag_00__CT_utility_damage_last_5s`: contribution `-0.001654`
- `lag_10__utility_damage_diff_last_5s`: contribution `-0.001483`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.001313`

### tick `132907`, seconds `52.00`, LSTM delta `+0.0632`

Top all feature movements:
- `lag_00__CT_place_BRICKS`: contribution `+0.015216`
- `lag_07__CT_place_BACKOFB`: contribution `+0.004645`
- `lag_09__CT_place_BACKOFB`: contribution `+0.004086`
- `lag_02__CT_place_BACKOFB`: contribution `+0.002349`
- `lag_03__T5__is_walking`: contribution `+0.001825`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131979`, seconds `37.50`, LSTM delta `-0.0629`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.012112`
- `lag_07__CT_place_BRIDGE`: contribution `-0.005477`
- `lag_00__CT1__shots_fired`: contribution `-0.004439`
- `lag_09__CT_place_BACKOFB`: contribution `-0.004086`
- `lag_01__CT_place_BRIDGE`: contribution `-0.003215`

Top utility-only movements:
- `lag_08__T2__molly`: contribution `-0.000471`

### tick `131947`, seconds `37.00`, LSTM delta `+0.0563`

Top all feature movements:
- `lag_06__CT_place_BRIDGE`: contribution `+0.009423`
- `lag_00__CT_place_BRIDGE`: contribution `+0.007977`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005047`
- `lag_08__CT_place_BACKOFB`: contribution `+0.002515`
- `lag_00__CT1__shots_fired`: contribution `+0.001850`

Top utility-only movements:
- `lag_03__T_active_infernos`: contribution `+0.000514`
- `lag_13__active_infernos_total`: contribution `+0.000454`
