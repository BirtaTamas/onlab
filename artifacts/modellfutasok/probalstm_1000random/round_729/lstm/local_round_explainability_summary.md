# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `5`

## Largest probability jumps

- tick `36941`, seconds `74.00`, LSTM `0.1805`, delta `-0.2853`
- tick `36749`, seconds `71.00`, LSTM `0.4255`, delta `-0.0830`
- tick `37517`, seconds `83.00`, LSTM `0.0286`, delta `-0.0768`
- tick `36557`, seconds `68.00`, LSTM `0.4447`, delta `-0.0729`
- tick `36589`, seconds `68.50`, LSTM `0.3759`, delta `-0.0688`
- tick `36909`, seconds `73.50`, LSTM `0.4658`, delta `+0.0526`
- tick `33549`, seconds `21.00`, LSTM `0.3454`, delta `-0.0504`
- tick `36365`, seconds `65.00`, LSTM `0.5260`, delta `+0.0484`
- tick `34189`, seconds `31.00`, LSTM `0.4777`, delta `+0.0475`
- tick `36653`, seconds `69.50`, LSTM `0.4594`, delta `+0.0463`

## Top 15 local ridge features

- `lag_12__T_place_SNIPERSNEST`: coefficient `-0.004856`, |coef| `0.004856`
- `lag_01__CT_place_SCAFFOLDING`: coefficient `0.002209`, |coef| `0.002209`
- `lag_00__CT2__is_walking`: coefficient `-0.001776`, |coef| `0.001776`
- `lag_10__CT1__is_scoped`: coefficient `-0.001636`, |coef| `0.001636`
- `lag_09__CT_place_STAIRS`: coefficient `-0.001528`, |coef| `0.001528`
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.001447`, |coef| `0.001447`
- `lag_13__T_place_SNIPERSNEST`: coefficient `-0.001446`, |coef| `0.001446`
- `lag_04__T_place_CONNECTOR`: coefficient `-0.001290`, |coef| `0.001290`
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_00__T_kills_last_3s`: coefficient `-0.001244`, |coef| `0.001244`
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `0.001132`, |coef| `0.001132`
- `lag_13__CT_place_STAIRS`: coefficient `-0.001129`, |coef| `0.001129`
- `lag_08__T5__is_scoped`: coefficient `-0.001118`, |coef| `0.001118`
- `lag_01__T_place_SNIPERSNEST`: coefficient `-0.001069`, |coef| `0.001069`
- `lag_09__CT3__is_walking`: coefficient `-0.001062`, |coef| `0.001062`

## Top 10 utility ridge features

- `lag_05__CT1__flash_duration`: coefficient `-0.000961` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000955` (raises CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.000769` (raises CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000757` (raises CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000728` (lowers CT win probability)
- `lag_15__CT1__flash_duration`: coefficient `0.000678` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.000604` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `0.000578` (raises CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000571` (raises CT win probability)
- `lag_00__CT_active_infernos`: coefficient `0.000558` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_SNIPERSNEST`: coefficient `-0.004856` (lowers CT win probability)
- `lag_01__CT_place_SCAFFOLDING`: coefficient `0.002209` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.001776` (lowers CT win probability)
- `lag_10__CT1__is_scoped`: coefficient `-0.001636` (lowers CT win probability)
- `lag_09__CT_place_STAIRS`: coefficient `-0.001528` (lowers CT win probability)
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.001447` (lowers CT win probability)
- `lag_13__T_place_SNIPERSNEST`: coefficient `-0.001446` (lowers CT win probability)
- `lag_04__T_place_CONNECTOR`: coefficient `-0.001290` (lowers CT win probability)
- `lag_00__CT_place_SCAFFOLDING`: coefficient `-0.001271` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001244` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `36941`, seconds `74.00`, LSTM delta `-0.2853`

Top all feature movements:
- `lag_12__T_place_SNIPERSNEST`: contribution `-0.086285`
- `lag_01__CT_place_SCAFFOLDING`: contribution `-0.046091`
- `lag_09__CT_place_STAIRS`: contribution `-0.011896`
- `lag_10__CT1__is_scoped`: contribution `-0.007007`
- `lag_08__T5__is_scoped`: contribution `-0.005331`

Top utility-only movements:
- `lag_05__CT1__flash_duration`: contribution `-0.004393`

### tick `36749`, seconds `71.00`, LSTM delta `-0.0830`

Top all feature movements:
- `lag_06__T_place_SNIPERSNEST`: contribution `-0.016769`
- `lag_15__CT_place_STAIRS`: contribution `-0.007200`
- `lag_03__CT_place_STAIRS`: contribution `-0.006422`
- `lag_15__T3__duck_amount`: contribution `-0.003607`
- `lag_06__CT_place_SHOP`: contribution `-0.003032`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002601`

### tick `37517`, seconds `83.00`, LSTM delta `-0.0768`

Top all feature movements:
- `lag_04__T_place_CONNECTOR`: contribution `-0.012495`
- `lag_07__CT_place_SHOP`: contribution `-0.005079`
- `lag_00__T_kills_last_3s`: contribution `-0.003940`
- `lag_15__CT1__flash_duration`: contribution `-0.003100`
- `lag_00__kill_diff_last_3s`: contribution `-0.002370`

Top utility-only movements:
- `lag_15__CT1__flash_duration`: contribution `-0.003100`

### tick `36557`, seconds `68.00`, LSTM delta `-0.0729`

Top all feature movements:
- `lag_00__T_place_SNIPERSNEST`: contribution `-0.025709`
- `lag_09__CT_place_STAIRS`: contribution `+0.011896`
- `lag_15__CT_place_STAIRS`: contribution `-0.007200`
- `lag_10__CT1__is_scoped`: contribution `-0.007007`
- `lag_15__CT_place_JUNGLE`: contribution `-0.005370`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `+0.001904`
- `lag_06__CT_B_site_active_infernos`: contribution `+0.001689`

### tick `36589`, seconds `68.50`, LSTM delta `-0.0688`

Top all feature movements:
- `lag_01__T_place_SNIPERSNEST`: contribution `-0.019001`
- `lag_13__CT_place_STAIRS`: contribution `-0.008785`
- `lag_01__CT_place_SHOP`: contribution `-0.003324`
- `lag_11__CT1__is_scoped`: contribution `-0.002653`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002601`

Top utility-only movements:
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002601`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.001348`
- `lag_00__CT_active_infernos`: contribution `-0.001287`
