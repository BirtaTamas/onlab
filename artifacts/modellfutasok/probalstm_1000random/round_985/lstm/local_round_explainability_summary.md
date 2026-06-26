# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `6`

## Largest probability jumps

- tick `42603`, seconds `46.50`, LSTM `0.0888`, delta `-0.2299`
- tick `42571`, seconds `46.00`, LSTM `0.3187`, delta `-0.1451`
- tick `41771`, seconds `33.50`, LSTM `0.4407`, delta `-0.0935`
- tick `41515`, seconds `29.50`, LSTM `0.4649`, delta `-0.0419`
- tick `41867`, seconds `35.00`, LSTM `0.4382`, delta `-0.0396`
- tick `41643`, seconds `31.50`, LSTM `0.5065`, delta `+0.0365`
- tick `41675`, seconds `32.00`, LSTM `0.5409`, delta `+0.0344`
- tick `39787`, seconds `2.50`, LSTM `0.4368`, delta `+0.0315`
- tick `42475`, seconds `44.50`, LSTM `0.4690`, delta `-0.0309`
- tick `40043`, seconds `6.50`, LSTM `0.4974`, delta `+0.0289`

## Top 15 local ridge features

- `lag_08__CT_place_SECRET`: coefficient `-0.002193`, |coef| `0.002193`
- `lag_07__CT_place_SECRET`: coefficient `-0.001564`, |coef| `0.001564`
- `lag_01__CT_place_SECRET`: coefficient `0.001443`, |coef| `0.001443`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001441`, |coef| `0.001441`
- `lag_04__CT_place_OBSERVATION`: coefficient `-0.001386`, |coef| `0.001386`
- `lag_02__CT_place_TUNNELS`: coefficient `-0.001316`, |coef| `0.001316`
- `lag_02__CT_place_SECRET`: coefficient `0.001210`, |coef| `0.001210`
- `lag_00__T1__shots_fired`: coefficient `-0.001137`, |coef| `0.001137`
- `lag_09__CT_place_SECRET`: coefficient `-0.001009`, |coef| `0.001009`
- `lag_03__CT_shots_fired_sum`: coefficient `-0.000975`, |coef| `0.000975`
- `lag_03__CT4__duck_amount`: coefficient `-0.000863`, |coef| `0.000863`
- `lag_15__CT_place_SECRET`: coefficient `-0.000856`, |coef| `0.000856`
- `lag_08__CT_place_OUTSIDE`: coefficient `0.000837`, |coef| `0.000837`
- `lag_03__CT_place_OBSERVATION`: coefficient `-0.000828`, |coef| `0.000828`
- `lag_08__T_place_RAMP`: coefficient `0.000810`, |coef| `0.000810`

## Top 10 utility ridge features

- `lag_00__T_smokes_last_5s`: coefficient `0.000607` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000593` (lowers CT win probability)
- `lag_04__T_B_site_active_infernos`: coefficient `-0.000562` (lowers CT win probability)
- `lag_07__CT5__molly`: coefficient `-0.000525` (lowers CT win probability)
- `lag_08__T1__molly`: coefficient `0.000451` (raises CT win probability)
- `lag_12__T2__smoke`: coefficient `0.000445` (raises CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.000430` (lowers CT win probability)
- `lag_01__T_utility_damage_last_5s`: coefficient `-0.000409` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000406` (lowers CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000403` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_08__CT_place_SECRET`: coefficient `-0.002193` (lowers CT win probability)
- `lag_07__CT_place_SECRET`: coefficient `-0.001564` (lowers CT win probability)
- `lag_01__CT_place_SECRET`: coefficient `0.001443` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001441` (lowers CT win probability)
- `lag_04__CT_place_OBSERVATION`: coefficient `-0.001386` (lowers CT win probability)
- `lag_02__CT_place_TUNNELS`: coefficient `-0.001316` (lowers CT win probability)
- `lag_02__CT_place_SECRET`: coefficient `0.001210` (raises CT win probability)
- `lag_00__T1__shots_fired`: coefficient `-0.001137` (lowers CT win probability)
- `lag_09__CT_place_SECRET`: coefficient `-0.001009` (lowers CT win probability)
- `lag_03__CT_shots_fired_sum`: coefficient `-0.000975` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `42603`, seconds `46.50`, LSTM delta `-0.2299`

Top all feature movements:
- `lag_04__CT_place_OBSERVATION`: contribution `-0.024137`
- `lag_08__CT_place_SECRET`: contribution `-0.022572`
- `lag_07__CT_place_SECRET`: contribution `-0.016097`
- `lag_01__CT_place_SECRET`: contribution `-0.014854`
- `lag_00__CT_place_OBSERVATION`: contribution `-0.013773`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `42571`, seconds `46.00`, LSTM delta `-0.1451`

Top all feature movements:
- `lag_08__CT_place_SECRET`: contribution `-0.022572`
- `lag_07__CT_place_SECRET`: contribution `-0.016097`
- `lag_01__CT_place_SECRET`: contribution `-0.014854`
- `lag_03__CT_place_OBSERVATION`: contribution `-0.014414`
- `lag_02__CT_place_SECRET`: contribution `-0.012456`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41771`, seconds `33.50`, LSTM delta `-0.0935`

Top all feature movements:
- `lag_08__CT_place_SECRET`: contribution `-0.022572`
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.006244`
- `lag_03__T_place_CONTROL`: contribution `-0.006231`
- `lag_14__T_place_TROPHY`: contribution `-0.005185`
- `lag_14__T_place_VENDING`: contribution `-0.005035`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41515`, seconds `29.50`, LSTM delta `-0.0419`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `-0.005118`
- `lag_06__T_place_VENDING`: contribution `-0.004868`
- `lag_09__CT_place_MINI`: contribution `-0.003238`
- `lag_04__CT_place_MINI`: contribution `-0.001954`
- `lag_02__T_place_SECRET`: contribution `-0.001888`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `41867`, seconds `35.00`, LSTM delta `-0.0396`

Top all feature movements:
- `lag_01__CT_place_SECRET`: contribution `-0.014854`
- `lag_03__CT_place_LOCKERROOM`: contribution `-0.006766`
- `lag_06__T_place_CONTROL`: contribution `-0.006123`
- `lag_11__T_place_TROPHY`: contribution `-0.003812`
- `lag_03__T_place_CONTROL`: contribution `+0.003115`

Top utility-only movements:
- No utility movement among the top local contributors.
