# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `14`

## Largest probability jumps

- tick `105960`, seconds `46.50`, LSTM `0.5757`, delta `+0.1971`
- tick `107752`, seconds `74.50`, LSTM `0.8672`, delta `+0.1792`
- tick `106120`, seconds `49.00`, LSTM `0.7922`, delta `+0.1747`
- tick `103016`, seconds `0.50`, LSTM `0.1672`, delta `-0.0664`
- tick `105864`, seconds `45.00`, LSTM `0.3378`, delta `-0.0604`
- tick `105192`, seconds `34.50`, LSTM `0.3544`, delta `+0.0567`
- tick `106408`, seconds `53.50`, LSTM `0.7795`, delta `+0.0551`
- tick `105896`, seconds `45.50`, LSTM `0.3896`, delta `+0.0518`
- tick `103976`, seconds `15.50`, LSTM `0.0992`, delta `-0.0491`
- tick `106312`, seconds `52.00`, LSTM `0.7343`, delta `-0.0435`

## Top 15 local ridge features

- `lag_05__T_place_LADDER`: coefficient `-0.002547`, |coef| `0.002547`
- `lag_00__CT_kills_last_3s`: coefficient `0.002467`, |coef| `0.002467`
- `lag_00__kill_diff_last_3s`: coefficient `0.002265`, |coef| `0.002265`
- `lag_12__CT_place_TSPAWN`: coefficient `0.002118`, |coef| `0.002118`
- `lag_04__CT_place_SIDEALLEY`: coefficient `-0.002053`, |coef| `0.002053`
- `lag_02__CT_place_JUNGLE`: coefficient `-0.002051`, |coef| `0.002051`
- `lag_10__CT_place_CATWALK`: coefficient `0.001958`, |coef| `0.001958`
- `lag_07__T1__is_walking`: coefficient `-0.001871`, |coef| `0.001871`
- `lag_07__CT_place_JUNGLE`: coefficient `0.001835`, |coef| `0.001835`
- `lag_00__T_place_CATWALK`: coefficient `-0.001751`, |coef| `0.001751`
- `lag_00__CT_damage_last_5s`: coefficient `0.001742`, |coef| `0.001742`
- `lag_10__CT_place_STAIRS`: coefficient `-0.001460`, |coef| `0.001460`
- `lag_00__damage_diff_last_5s`: coefficient `0.001434`, |coef| `0.001434`
- `lag_10__CT_place_TOPOFMID`: coefficient `-0.001431`, |coef| `0.001431`
- `lag_08__CT3__duck_amount`: coefficient `0.001403`, |coef| `0.001403`

## Top 10 utility ridge features

- `lag_10__T1__smoke`: coefficient `-0.000899` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000788` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.000746` (lowers CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `-0.000713` (lowers CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000708` (lowers CT win probability)
- `lag_01__T3__molly`: coefficient `-0.000603` (lowers CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `-0.000544` (lowers CT win probability)
- `lag_12__active_infernos_total`: coefficient `-0.000542` (lowers CT win probability)
- `lag_01__T3__utility_total`: coefficient `-0.000539` (lowers CT win probability)
- `lag_01__CT4__smoke`: coefficient `-0.000529` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_LADDER`: coefficient `-0.002547` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002467` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002265` (raises CT win probability)
- `lag_12__CT_place_TSPAWN`: coefficient `0.002118` (raises CT win probability)
- `lag_04__CT_place_SIDEALLEY`: coefficient `-0.002053` (lowers CT win probability)
- `lag_02__CT_place_JUNGLE`: coefficient `-0.002051` (lowers CT win probability)
- `lag_10__CT_place_CATWALK`: coefficient `0.001958` (raises CT win probability)
- `lag_07__T1__is_walking`: coefficient `-0.001871` (lowers CT win probability)
- `lag_07__CT_place_JUNGLE`: coefficient `0.001835` (raises CT win probability)
- `lag_00__T_place_CATWALK`: coefficient `-0.001751` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `105960`, seconds `46.50`, LSTM delta `+0.1971`

Top all feature movements:
- `lag_04__T_place_LADDER`: contribution `+0.025974`
- `lag_00__T_place_LADDER`: contribution `+0.024869`
- `lag_12__CT_place_TSPAWN`: contribution `+0.015853`
- `lag_00__CT_kills_last_3s`: contribution `+0.007124`
- `lag_03__CT_place_TRUCK`: contribution `+0.005708`

Top utility-only movements:
- `lag_12__T_B_site_active_infernos`: contribution `+0.002229`

### tick `107752`, seconds `74.50`, LSTM delta `+0.1792`

Top all feature movements:
- `lag_05__T_place_LADDER`: contribution `+0.057572`
- `lag_02__CT_place_JUNGLE`: contribution `+0.013157`
- `lag_07__CT_place_JUNGLE`: contribution `+0.011771`
- `lag_10__CT_place_CATWALK`: contribution `+0.007800`
- `lag_00__CT_kills_last_3s`: contribution `+0.007124`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106120`, seconds `49.00`, LSTM delta `+0.1747`

Top all feature movements:
- `lag_05__T_place_LADDER`: contribution `+0.057572`
- `lag_00__CT_place_SIDEALLEY`: contribution `+0.016285`
- `lag_09__T_place_LADDER`: contribution `+0.014242`
- `lag_08__CT_place_TRUCK`: contribution `+0.008279`
- `lag_00__CT_kills_last_3s`: contribution `+0.007124`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103016`, seconds `0.50`, LSTM delta `-0.0664`

Top all feature movements:
- `lag_01__T_place_TSPAWN`: contribution `-0.004840`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.004659`
- `lag_00__CT_velocity_mean`: contribution `-0.003543`
- `lag_00__T_velocity_mean`: contribution `-0.002464`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.002258`

Top utility-only movements:
- `lag_01__T_smoke_inv`: contribution `-0.001614`
- `lag_01__T3__molly`: contribution `-0.000961`
- `lag_01__T3__utility_total`: contribution `-0.000848`
- `lag_01__CT4__smoke`: contribution `-0.000808`
- `lag_01__T1__molly`: contribution `-0.000808`

### tick `105864`, seconds `45.00`, LSTM delta `-0.0604`

Top all feature movements:
- `lag_09__CT_place_TSPAWN`: contribution `-0.005239`
- `lag_08__CT3__duck_amount`: contribution `-0.005219`
- `lag_00__CT1__is_walking`: contribution `-0.003007`
- `lag_15__CT_place_JUNGLE`: contribution `+0.002979`
- `lag_11__CT1__is_walking`: contribution `-0.002735`

Top utility-only movements:
- No utility movement among the top local contributors.
