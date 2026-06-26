# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `6`

## Largest probability jumps

- tick `27115`, seconds `35.00`, LSTM `0.8986`, delta `+0.2530`
- tick `27083`, seconds `34.50`, LSTM `0.6456`, delta `+0.0401`
- tick `28043`, seconds `49.50`, LSTM `0.9388`, delta `+0.0388`
- tick `24939`, seconds `1.00`, LSTM `0.6078`, delta `-0.0285`
- tick `25483`, seconds `9.50`, LSTM `0.5952`, delta `+0.0273`
- tick `28139`, seconds `51.00`, LSTM `0.9737`, delta `+0.0260`
- tick `27051`, seconds `34.00`, LSTM `0.6054`, delta `+0.0214`
- tick `27787`, seconds `45.50`, LSTM `0.8909`, delta `+0.0208`
- tick `27147`, seconds `35.50`, LSTM `0.9171`, delta `+0.0185`
- tick `26795`, seconds `30.00`, LSTM `0.5813`, delta `+0.0174`

## Top 15 local ridge features

- `lag_01__CT_shots_fired_sum`: coefficient `0.002118`, |coef| `0.002118`
- `lag_02__CT_shots_fired_sum`: coefficient `0.001783`, |coef| `0.001783`
- `lag_00__CT_kills_last_3s`: coefficient `0.001589`, |coef| `0.001589`
- `lag_00__kill_diff_last_3s`: coefficient `0.001325`, |coef| `0.001325`
- `lag_00__damage_diff_last_5s`: coefficient `0.001274`, |coef| `0.001274`
- `lag_00__CT_damage_last_5s`: coefficient `0.001221`, |coef| `0.001221`
- `lag_01__T4__shots_fired`: coefficient `0.001217`, |coef| `0.001217`
- `lag_00__CT1__shots_fired`: coefficient `-0.001057`, |coef| `0.001057`
- `lag_10__T2__duck_amount`: coefficient `-0.001052`, |coef| `0.001052`
- `lag_02__CT_place_TSIDEUPPER`: coefficient `0.001043`, |coef| `0.001043`
- `lag_03__CT_shots_fired_sum`: coefficient `0.000964`, |coef| `0.000964`
- `lag_00__T4__shots_fired`: coefficient `0.000952`, |coef| `0.000952`
- `lag_14__CT5__flash_duration`: coefficient `-0.000926`, |coef| `0.000926`
- `lag_06__T_shots_fired_sum`: coefficient `0.000914`, |coef| `0.000914`
- `lag_00__closest_enemy_dist_diff`: coefficient `0.000910`, |coef| `0.000910`

## Top 10 utility ridge features

- `lag_14__CT5__flash_duration`: coefficient `-0.000926` (lowers CT win probability)
- `lag_00__T3__molly`: coefficient `-0.000805` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000788` (lowers CT win probability)
- `lag_08__CT1__smoke`: coefficient `-0.000665` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000646` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000533` (lowers CT win probability)
- `lag_02__CT_utility_damage_last_5s`: coefficient `0.000528` (raises CT win probability)
- `lag_10__CT_B_site_active_smokes`: coefficient `-0.000518` (lowers CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000492` (lowers CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `-0.000455` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_shots_fired_sum`: coefficient `0.002118` (raises CT win probability)
- `lag_02__CT_shots_fired_sum`: coefficient `0.001783` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001589` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001325` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001274` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001221` (raises CT win probability)
- `lag_01__T4__shots_fired`: coefficient `0.001217` (raises CT win probability)
- `lag_00__CT1__shots_fired`: coefficient `-0.001057` (lowers CT win probability)
- `lag_10__T2__duck_amount`: coefficient `-0.001052` (lowers CT win probability)
- `lag_02__CT_place_TSIDEUPPER`: coefficient `0.001043` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `27115`, seconds `35.00`, LSTM delta `+0.2530`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.016184`
- `lag_00__CT_kills_last_3s`: contribution `+0.009176`
- `lag_02__CT_shots_fired_sum`: contribution `+0.008670`
- `lag_02__CT_place_TSIDEUPPER`: contribution `+0.007843`
- `lag_00__CT1__shots_fired`: contribution `+0.006703`

Top utility-only movements:
- `lag_14__CT5__flash_duration`: contribution `+0.004361`

### tick `27083`, seconds `34.50`, LSTM delta `+0.0401`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.010299`
- `lag_00__T_shots_fired_sum`: contribution `-0.003287`
- `lag_00__T4__shots_fired`: contribution `+0.002939`
- `lag_00__CT1__shots_fired`: contribution `-0.002793`
- `lag_03__T_shots_fired_sum`: contribution `-0.002675`

Top utility-only movements:
- `lag_13__CT5__flash_duration`: contribution `+0.002143`

### tick `28043`, seconds `49.50`, LSTM delta `+0.0388`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.004588`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004414`
- `lag_00__kill_diff_last_3s`: contribution `+0.003189`
- `lag_08__CT_place_TSIDELOWER`: contribution `+0.002246`
- `lag_00__damage_diff_last_5s`: contribution `+0.002069`

Top utility-only movements:
- `lag_04__T1__flash_duration`: contribution `+0.000822`
- `lag_04__CT5__flash_duration`: contribution `-0.000803`
- `lag_04__T2__flash_duration`: contribution `+0.000750`

### tick `24939`, seconds `1.00`, LSTM delta `-0.0285`

Top all feature movements:
- `lag_02__CT_place_MAINHALL`: contribution `-0.002590`
- `lag_01__CT_velocity_mean`: contribution `-0.001400`
- `lag_02__T_place_TSPAWN`: contribution `-0.001203`
- `lag_02__T_closest_enemy_dist`: contribution `-0.000838`
- `lag_01__CT3__is_walking`: contribution `-0.000678`

Top utility-only movements:
- `lag_02__T_smoke_inv`: contribution `-0.000557`
- `lag_02__CT_smoke_inv`: contribution `-0.000530`
- `lag_02__T_utility_inv`: contribution `-0.000511`
- `lag_02__CT1__utility_total`: contribution `-0.000501`
- `lag_02__T3__molly`: contribution `-0.000440`

### tick `25483`, seconds `9.50`, LSTM delta `+0.0273`

Top all feature movements:
- `lag_11__CT_place_HOUSE`: contribution `+0.004680`
- `lag_00__T_place_TSIDELOWER`: contribution `-0.003306`
- `lag_14__CT_place_MAINHALL`: contribution `+0.003086`
- `lag_00__T_place_TUNNEL`: contribution `+0.002103`
- `lag_00__T_place_WATER`: contribution `+0.002002`

Top utility-only movements:
- `lag_01__CT_B_site_active_infernos`: contribution `+0.001229`
- `lag_02__utility_damage_diff_last_5s`: contribution `-0.000851`
- `lag_01__CT_active_infernos`: contribution `+0.000521`
