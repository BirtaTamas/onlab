# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `41106`, seconds `37.00`, LSTM `0.8150`, delta `+0.1319`
- tick `41490`, seconds `43.00`, LSTM `0.9506`, delta `+0.0570`
- tick `41170`, seconds `38.00`, LSTM `0.9165`, delta `+0.0517`
- tick `41138`, seconds `37.50`, LSTM `0.8648`, delta `+0.0497`
- tick `39634`, seconds `14.00`, LSTM `0.6957`, delta `-0.0348`
- tick `39122`, seconds `6.00`, LSTM `0.7389`, delta `-0.0295`
- tick `39442`, seconds `11.00`, LSTM `0.7359`, delta `-0.0222`
- tick `40466`, seconds `27.00`, LSTM `0.6617`, delta `+0.0209`
- tick `38770`, seconds `0.50`, LSTM `0.7913`, delta `+0.0185`
- tick `39378`, seconds `10.00`, LSTM `0.7574`, delta `+0.0178`

## Top 15 local ridge features

- `lag_07__T_place_SIDEENTRANCE`: coefficient `0.001513`, |coef| `0.001513`
- `lag_08__T_place_SIDEENTRANCE`: coefficient `0.001336`, |coef| `0.001336`
- `lag_00__CT3__shots_fired`: coefficient `0.001325`, |coef| `0.001325`
- `lag_14__CT_place_SIDEENTRANCE`: coefficient `0.001304`, |coef| `0.001304`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001190`, |coef| `0.001190`
- `lag_05__T_place_SIDEENTRANCE`: coefficient `0.001055`, |coef| `0.001055`
- `lag_03__CT3__duck_amount`: coefficient `0.000992`, |coef| `0.000992`
- `lag_00__CT_kills_last_3s`: coefficient `0.000984`, |coef| `0.000984`
- `lag_09__T_place_SIDEENTRANCE`: coefficient `0.000971`, |coef| `0.000971`
- `lag_00__kill_diff_last_3s`: coefficient `0.000835`, |coef| `0.000835`
- `lag_09__T_place_TSIDELOWER`: coefficient `-0.000834`, |coef| `0.000834`
- `lag_12__T3__duck_amount`: coefficient `-0.000817`, |coef| `0.000817`
- `lag_07__T_B_site_active_infernos`: coefficient `0.000799`, |coef| `0.000799`
- `lag_00__T4__alive`: coefficient `-0.000796`, |coef| `0.000796`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000790`, |coef| `0.000790`

## Top 10 utility ridge features

- `lag_07__T_B_site_active_infernos`: coefficient `0.000799` (raises CT win probability)
- `lag_14__CT3__smoke`: coefficient `-0.000706` (lowers CT win probability)
- `lag_11__T1__molly`: coefficient `-0.000703` (lowers CT win probability)
- `lag_02__T1__smoke`: coefficient `-0.000697` (lowers CT win probability)
- `lag_11__CT1__smoke`: coefficient `-0.000689` (lowers CT win probability)
- `lag_10__CT3__smoke`: coefficient `0.000596` (raises CT win probability)
- `lag_07__T_active_infernos`: coefficient `0.000595` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.000568` (lowers CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `0.000495` (raises CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `0.000484` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_07__T_place_SIDEENTRANCE`: coefficient `0.001513` (raises CT win probability)
- `lag_08__T_place_SIDEENTRANCE`: coefficient `0.001336` (raises CT win probability)
- `lag_00__CT3__shots_fired`: coefficient `0.001325` (raises CT win probability)
- `lag_14__CT_place_SIDEENTRANCE`: coefficient `0.001304` (raises CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001190` (lowers CT win probability)
- `lag_05__T_place_SIDEENTRANCE`: coefficient `0.001055` (raises CT win probability)
- `lag_03__CT3__duck_amount`: coefficient `0.000992` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000984` (raises CT win probability)
- `lag_09__T_place_SIDEENTRANCE`: coefficient `0.000971` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000835` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `41106`, seconds `37.00`, LSTM delta `+0.1319`

Top all feature movements:
- `lag_07__T_place_SIDEENTRANCE`: contribution `+0.007385`
- `lag_08__T_place_SIDEENTRANCE`: contribution `+0.006521`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.005806`
- `lag_14__CT_place_SIDEENTRANCE`: contribution `+0.005251`
- `lag_05__T_place_SIDEENTRANCE`: contribution `+0.005150`

Top utility-only movements:
- `lag_07__T_B_site_active_infernos`: contribution `+0.002258`
- `lag_14__CT3__smoke`: contribution `+0.001561`

### tick `41490`, seconds `43.00`, LSTM delta `+0.0570`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.005806`
- `lag_08__CT3__shots_fired`: contribution `+0.005010`
- `lag_07__CT2__is_scoped`: contribution `+0.003045`
- `lag_00__CT_kills_last_3s`: contribution `+0.002840`
- `lag_05__CT1__flash_duration`: contribution `+0.002653`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `+0.002653`

### tick `41170`, seconds `38.00`, LSTM delta `+0.0517`

Top all feature movements:
- `lag_07__T_place_SIDEENTRANCE`: contribution `+0.007385`
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.005806`
- `lag_09__T_place_SIDEENTRANCE`: contribution `+0.004737`
- `lag_00__CT3__shots_fired`: contribution `+0.003407`
- `lag_00__CT_kills_last_3s`: contribution `+0.002840`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41138`, seconds `37.50`, LSTM delta `+0.0497`

Top all feature movements:
- `lag_08__T_place_SIDEENTRANCE`: contribution `+0.006521`
- `lag_00__T_place_SIDEENTRANCE`: contribution `-0.005806`
- `lag_09__T_place_SIDEENTRANCE`: contribution `+0.004737`
- `lag_00__CT3__shots_fired`: contribution `+0.003407`
- `lag_15__CT_place_SIDEENTRANCE`: contribution `+0.002806`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `+0.000965`
- `lag_08__T_B_site_active_infernos`: contribution `+0.000908`

### tick `39634`, seconds `14.00`, LSTM delta `-0.0348`

Top all feature movements:
- `lag_00__T_mollies_last_5s`: contribution `-0.010179`
- `lag_10__T_mollies_last_5s`: contribution `-0.004883`
- `lag_02__T_place_WATER`: contribution `-0.002266`
- `lag_06__T_flashes_last_5s`: contribution `-0.001967`
- `lag_13__T3__duck_amount`: contribution `+0.001525`

Top utility-only movements:
- `lag_00__T_mollies_last_5s`: contribution `-0.010179`
- `lag_10__T_mollies_last_5s`: contribution `-0.004883`
- `lag_06__T_flashes_last_5s`: contribution `-0.001967`
