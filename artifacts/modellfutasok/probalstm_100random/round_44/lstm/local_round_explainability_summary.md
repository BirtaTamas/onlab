# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `10`

## Largest probability jumps

- tick `96763`, seconds `78.50`, LSTM `0.9099`, delta `+0.1181`
- tick `93307`, seconds `24.50`, LSTM `0.8133`, delta `+0.0721`
- tick `96827`, seconds `79.50`, LSTM `0.9710`, delta `+0.0494`
- tick `93403`, seconds `26.00`, LSTM `0.8598`, delta `+0.0382`
- tick `93979`, seconds `35.00`, LSTM `0.8535`, delta `-0.0353`
- tick `95003`, seconds `51.00`, LSTM `0.8420`, delta `+0.0289`
- tick `94523`, seconds `43.50`, LSTM `0.8468`, delta `-0.0278`
- tick `93883`, seconds `33.50`, LSTM `0.8851`, delta `+0.0261`
- tick `93627`, seconds `29.50`, LSTM `0.8548`, delta `+0.0254`
- tick `93275`, seconds `24.00`, LSTM `0.7412`, delta `+0.0253`

## Top 15 local ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001747`, |coef| `0.001747`
- `lag_00__kill_diff_last_3s`: coefficient `0.001457`, |coef| `0.001457`
- `lag_00__T4__is_walking`: coefficient `-0.001425`, |coef| `0.001425`
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001269`, |coef| `0.001269`
- `lag_04__CT3__is_scoped`: coefficient `-0.001267`, |coef| `0.001267`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001240`, |coef| `0.001240`
- `lag_14__T_place_SIDEENTRANCE`: coefficient `0.001172`, |coef| `0.001172`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001058`, |coef| `0.001058`
- `lag_12__T2__is_walking`: coefficient `0.000997`, |coef| `0.000997`
- `lag_00__T_walking_count`: coefficient `-0.000953`, |coef| `0.000953`
- `lag_00__CT3__molly`: coefficient `-0.000929`, |coef| `0.000929`
- `lag_00__T4__alive`: coefficient `-0.000925`, |coef| `0.000925`
- `lag_00__CT_damage_last_5s`: coefficient `0.000893`, |coef| `0.000893`
- `lag_15__T_place_SIDEENTRANCE`: coefficient `0.000855`, |coef| `0.000855`
- `lag_00__T4__armor`: coefficient `-0.000832`, |coef| `0.000832`

## Top 10 utility ridge features

- `lag_00__CT3__molly`: coefficient `-0.000929` (lowers CT win probability)
- `lag_14__T2__smoke`: coefficient `-0.000741` (lowers CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000713` (lowers CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000666` (lowers CT win probability)
- `lag_15__T_B_site_active_infernos`: coefficient `-0.000629` (lowers CT win probability)
- `lag_08__T_B_site_active_infernos`: coefficient `0.000529` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000525` (lowers CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.000504` (lowers CT win probability)
- `lag_15__T_active_infernos`: coefficient `-0.000472` (lowers CT win probability)
- `lag_08__T_A_site_active_smokes`: coefficient `0.000458` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001747` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001457` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.001425` (lowers CT win probability)
- `lag_00__T_place_SIDEENTRANCE`: coefficient `-0.001269` (lowers CT win probability)
- `lag_04__CT3__is_scoped`: coefficient `-0.001267` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001240` (raises CT win probability)
- `lag_14__T_place_SIDEENTRANCE`: coefficient `0.001172` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001058` (raises CT win probability)
- `lag_12__T2__is_walking`: coefficient `0.000997` (raises CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000953` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `96763`, seconds `78.50`, LSTM delta `+0.1181`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.006193`
- `lag_04__CT3__is_scoped`: contribution `+0.005762`
- `lag_14__T_place_SIDEENTRANCE`: contribution `+0.005720`
- `lag_00__CT_kills_last_3s`: contribution `+0.005044`
- `lag_00__kill_diff_last_3s`: contribution `+0.003506`

Top utility-only movements:
- `lag_00__CT3__molly`: contribution `+0.002294`
- `lag_15__T_B_site_active_infernos`: contribution `+0.001778`
- `lag_14__T2__smoke`: contribution `+0.001627`

### tick `93307`, seconds `24.50`, LSTM delta `+0.0721`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.005044`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004307`
- `lag_15__T_place_TUNNEL`: contribution `+0.004300`
- `lag_00__kill_diff_last_3s`: contribution `+0.003506`
- `lag_00__T4__is_walking`: contribution `+0.003289`

Top utility-only movements:
- `lag_09__CT_B_site_active_infernos`: contribution `+0.001397`

### tick `96827`, seconds `79.50`, LSTM delta `+0.0494`

Top all feature movements:
- `lag_00__T_place_SIDEENTRANCE`: contribution `+0.006193`
- `lag_00__CT_kills_last_3s`: contribution `+0.005044`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004307`
- `lag_00__kill_diff_last_3s`: contribution `+0.003506`
- `lag_02__T_place_SIDEENTRANCE`: contribution `+0.002660`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `93403`, seconds `26.00`, LSTM delta `+0.0382`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004307`
- `lag_10__T_place_TSIDELOWER`: contribution `+0.001981`
- `lag_00__T_place_TSIDELOWER`: contribution `+0.001757`
- `lag_09__T_place_TSIDELOWER`: contribution `-0.001621`
- `lag_09__T4__is_walking`: contribution `+0.001457`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `93979`, seconds `35.00`, LSTM delta `-0.0353`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.007951`
- `lag_00__T4__is_walking`: contribution `-0.003289`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002354`
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.002058`
- `lag_03__CT_place_SIDEENTRANCE`: contribution `-0.001625`

Top utility-only movements:
- No utility movement among the top local contributors.
