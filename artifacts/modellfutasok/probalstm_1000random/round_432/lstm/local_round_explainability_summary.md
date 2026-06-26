# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `2`

## Largest probability jumps

- tick `9962`, seconds `28.50`, LSTM `0.8930`, delta `+0.2175`
- tick `9866`, seconds `27.00`, LSTM `0.6581`, delta `+0.1570`
- tick `10058`, seconds `30.00`, LSTM `0.9552`, delta `+0.0680`
- tick `9930`, seconds `28.00`, LSTM `0.6755`, delta `+0.0585`
- tick `9898`, seconds `27.50`, LSTM `0.6170`, delta `-0.0411`
- tick `10570`, seconds `38.00`, LSTM `0.9592`, delta `+0.0393`
- tick `9290`, seconds `18.00`, LSTM `0.5878`, delta `+0.0211`
- tick `9706`, seconds `24.50`, LSTM `0.5352`, delta `-0.0205`
- tick `9834`, seconds `26.50`, LSTM `0.5011`, delta `-0.0203`
- tick `9002`, seconds `13.50`, LSTM `0.5734`, delta `+0.0195`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001591`, |coef| `0.001591`
- `lag_08__CT_place_STAIRS`: coefficient `-0.001216`, |coef| `0.001216`
- `lag_04__CT_place_UNDERPASS`: coefficient `0.001215`, |coef| `0.001215`
- `lag_01__CT_place_UNDERPASS`: coefficient `0.001193`, |coef| `0.001193`
- `lag_05__CT_place_STAIRS`: coefficient `-0.001058`, |coef| `0.001058`
- `lag_09__CT_place_TRUCK`: coefficient `-0.001055`, |coef| `0.001055`
- `lag_06__CT_place_JUNGLE`: coefficient `0.001030`, |coef| `0.001030`
- `lag_00__CT_kills_last_3s`: coefficient `0.001008`, |coef| `0.001008`
- `lag_00__CT2__shots_fired`: coefficient `0.000968`, |coef| `0.000968`
- `lag_03__CT2__shots_fired`: coefficient `0.000961`, |coef| `0.000961`
- `lag_09__CT_place_UNDERPASS`: coefficient `-0.000960`, |coef| `0.000960`
- `lag_06__CT3__flash_duration`: coefficient `0.000952`, |coef| `0.000952`
- `lag_13__T_place_BACKALLEY`: coefficient `-0.000948`, |coef| `0.000948`
- `lag_06__CT_place_TRUCK`: coefficient `-0.000918`, |coef| `0.000918`
- `lag_00__CT_damage_last_5s`: coefficient `0.000912`, |coef| `0.000912`

## Top 10 utility ridge features

- `lag_06__CT3__flash_duration`: coefficient `0.000952` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.000782` (raises CT win probability)
- `lag_05__CT_B_site_active_infernos`: coefficient `0.000608` (raises CT win probability)
- `lag_05__CT3__flash_duration`: coefficient `0.000578` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000559` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000550` (raises CT win probability)
- `lag_06__CT1__molly`: coefficient `-0.000501` (lowers CT win probability)
- `lag_10__CT2__smoke`: coefficient `-0.000480` (lowers CT win probability)
- `lag_05__CT_active_infernos`: coefficient `0.000439` (raises CT win probability)
- `lag_14__T2__smoke`: coefficient `-0.000439` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001591` (raises CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `-0.001216` (lowers CT win probability)
- `lag_04__CT_place_UNDERPASS`: coefficient `0.001215` (raises CT win probability)
- `lag_01__CT_place_UNDERPASS`: coefficient `0.001193` (raises CT win probability)
- `lag_05__CT_place_STAIRS`: coefficient `-0.001058` (lowers CT win probability)
- `lag_09__CT_place_TRUCK`: coefficient `-0.001055` (lowers CT win probability)
- `lag_06__CT_place_JUNGLE`: coefficient `0.001030` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001008` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.000968` (raises CT win probability)
- `lag_03__CT2__shots_fired`: coefficient `0.000961` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `9962`, seconds `28.50`, LSTM delta `+0.2175`

Top all feature movements:
- `lag_08__CT_place_STAIRS`: contribution `+0.009468`
- `lag_04__CT_place_UNDERPASS`: contribution `+0.007043`
- `lag_09__CT_place_TRUCK`: contribution `+0.006808`
- `lag_06__CT_place_JUNGLE`: contribution `+0.006608`
- `lag_03__CT_shots_fired_sum`: contribution `+0.006515`

Top utility-only movements:
- `lag_06__CT3__flash_duration`: contribution `+0.004775`

### tick `9866`, seconds `27.00`, LSTM delta `+0.1570`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.012156`
- `lag_05__CT_place_STAIRS`: contribution `+0.008235`
- `lag_01__CT_place_UNDERPASS`: contribution `+0.006916`
- `lag_06__CT_place_TRUCK`: contribution `+0.005922`
- `lag_06__CT_place_UNDERPASS`: contribution `+0.004101`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `+0.003924`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.001889`

### tick `10058`, seconds `30.00`, LSTM delta `+0.0680`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.008841`
- `lag_00__CT_place_SCAFFOLDING`: contribution `+0.008271`
- `lag_01__T_shots_fired_sum`: contribution `+0.005639`
- `lag_02__T_place_TRUCK`: contribution `+0.005458`
- `lag_05__CT_shots_fired_sum`: contribution `-0.005349`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `9930`, seconds `28.00`, LSTM delta `+0.0585`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007736`
- `lag_03__CT_place_UNDERPASS`: contribution `+0.003792`
- `lag_03__CT_shots_fired_sum`: contribution `-0.002961`
- `lag_05__CT3__flash_duration`: contribution `+0.002900`
- `lag_02__CT_shots_fired_sum`: contribution `-0.002504`

Top utility-only movements:
- `lag_05__CT3__flash_duration`: contribution `+0.002900`

### tick `9898`, seconds `27.50`, LSTM delta `-0.0411`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.014366`
- `lag_00__CT2__shots_fired`: contribution `-0.004330`
- `lag_03__CT_shots_fired_sum`: contribution `+0.003553`
- `lag_02__T_flashed_players`: contribution `-0.003221`
- `lag_04__T4__duck_amount`: contribution `-0.002781`

Top utility-only movements:
- No utility movement among the top local contributors.
