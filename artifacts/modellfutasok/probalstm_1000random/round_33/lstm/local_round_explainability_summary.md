# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `76721`, seconds `82.50`, LSTM `0.6100`, delta `+0.3502`
- tick `74257`, seconds `44.00`, LSTM `0.6284`, delta `+0.2431`
- tick `75505`, seconds `63.50`, LSTM `0.5141`, delta `-0.2143`
- tick `73201`, seconds `27.50`, LSTM `0.4882`, delta `-0.1429`
- tick `73009`, seconds `24.50`, LSTM `0.6789`, delta `+0.1411`
- tick `73969`, seconds `39.50`, LSTM `0.4919`, delta `-0.1225`
- tick `75601`, seconds `65.00`, LSTM `0.2826`, delta `-0.1145`
- tick `73745`, seconds `36.00`, LSTM `0.6049`, delta `+0.1114`
- tick `73713`, seconds `35.50`, LSTM `0.4935`, delta `+0.1018`
- tick `75569`, seconds `64.50`, LSTM `0.3971`, delta `-0.0847`

## Top 15 local ridge features

- `lag_12__T_place_CONNECTOR`: coefficient `-0.005960`, |coef| `0.005960`
- `lag_00__kill_diff_last_3s`: coefficient `0.005788`, |coef| `0.005788`
- `lag_12__T_place_JUNGLE`: coefficient `0.005370`, |coef| `0.005370`
- `lag_00__damage_diff_last_5s`: coefficient `0.005021`, |coef| `0.005021`
- `lag_00__CT_kills_last_3s`: coefficient `0.004636`, |coef| `0.004636`
- `lag_00__CT_velocity_mean`: coefficient `-0.003915`, |coef| `0.003915`
- `lag_01__CT_place_TRUCK`: coefficient `-0.003230`, |coef| `0.003230`
- `lag_00__CT_damage_last_5s`: coefficient `0.003003`, |coef| `0.003003`
- `lag_14__CT_place_SHOP`: coefficient `-0.002866`, |coef| `0.002866`
- `lag_00__T5__is_scoped`: coefficient `0.002847`, |coef| `0.002847`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002846`, |coef| `0.002846`
- `lag_10__T2__duck_amount`: coefficient `0.002777`, |coef| `0.002777`
- `lag_12__CT4__is_walking`: coefficient `-0.002586`, |coef| `0.002586`
- `lag_00__T_kills_last_3s`: coefficient `-0.002531`, |coef| `0.002531`
- `lag_07__T_place_PALACEALLEY`: coefficient `0.002284`, |coef| `0.002284`

## Top 10 utility ridge features

- `lag_00__CT4__smoke`: coefficient `0.001388` (raises CT win probability)
- `lag_04__T2__molly`: coefficient `-0.001255` (lowers CT win probability)
- `lag_04__T5__smoke`: coefficient `-0.001222` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001207` (raises CT win probability)
- `lag_09__CT2__smoke`: coefficient `-0.001192` (lowers CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001103` (raises CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.001090` (lowers CT win probability)
- `lag_00__T_A_site_active_infernos`: coefficient `-0.001051` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001020` (lowers CT win probability)
- `lag_12__T_A_site_active_infernos`: coefficient `0.000985` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_CONNECTOR`: coefficient `-0.005960` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005788` (raises CT win probability)
- `lag_12__T_place_JUNGLE`: coefficient `0.005370` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.005021` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004636` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.003915` (lowers CT win probability)
- `lag_01__CT_place_TRUCK`: coefficient `-0.003230` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.003003` (raises CT win probability)
- `lag_14__CT_place_SHOP`: coefficient `-0.002866` (lowers CT win probability)
- `lag_00__T5__is_scoped`: coefficient `0.002847` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `76721`, seconds `82.50`, LSTM delta `+0.3502`

Top all feature movements:
- `lag_12__T_place_JUNGLE`: contribution `+0.069557`
- `lag_12__T_place_CONNECTOR`: contribution `+0.028862`
- `lag_14__CT_place_SHOP`: contribution `+0.014375`
- `lag_00__kill_diff_last_3s`: contribution `+0.013931`
- `lag_00__CT_kills_last_3s`: contribution `+0.013385`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `74257`, seconds `44.00`, LSTM delta `+0.2431`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013931`
- `lag_00__CT_kills_last_3s`: contribution `+0.013385`
- `lag_00__damage_diff_last_5s`: contribution `+0.011328`
- `lag_10__T2__duck_amount`: contribution `+0.010617`
- `lag_00__CT_shots_fired_sum`: contribution `+0.009887`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `75505`, seconds `63.50`, LSTM delta `-0.2143`

Top all feature movements:
- `lag_12__T_place_CONNECTOR`: contribution `-0.028862`
- `lag_01__CT_place_TRUCK`: contribution `-0.020837`
- `lag_00__kill_diff_last_3s`: contribution `-0.013931`
- `lag_00__T5__is_scoped`: contribution `-0.013579`
- `lag_00__damage_diff_last_5s`: contribution `-0.011328`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73201`, seconds `27.50`, LSTM delta `-0.1429`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.027863`
- `lag_00__T5__is_scoped`: contribution `-0.013579`
- `lag_00__CT_kills_last_3s`: contribution `-0.013385`
- `lag_00__damage_diff_last_5s`: contribution `-0.011328`
- `lag_00__T_kills_last_3s`: contribution `-0.008018`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `73009`, seconds `24.50`, LSTM delta `+0.1411`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.013931`
- `lag_00__CT_shots_fired_sum`: contribution `+0.013842`
- `lag_00__CT_kills_last_3s`: contribution `+0.013385`
- `lag_00__damage_diff_last_5s`: contribution `+0.008383`
- `lag_00__CT_damage_last_5s`: contribution `+0.006545`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.003018`
- `lag_14__T5__flash_duration`: contribution `+0.001958`
- `lag_15__CT2__flash_duration`: contribution `+0.001610`
