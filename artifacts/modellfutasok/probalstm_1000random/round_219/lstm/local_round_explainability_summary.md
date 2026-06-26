# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `73709`, seconds `39.00`, LSTM `0.7614`, delta `+0.1302`
- tick `73741`, seconds `39.50`, LSTM `0.8706`, delta `+0.1092`
- tick `74669`, seconds `54.00`, LSTM `0.9114`, delta `+0.0527`
- tick `73325`, seconds `33.00`, LSTM `0.6263`, delta `+0.0237`
- tick `73293`, seconds `32.50`, LSTM `0.6026`, delta `-0.0222`
- tick `71821`, seconds `9.50`, LSTM `0.5992`, delta `-0.0218`
- tick `72717`, seconds `23.50`, LSTM `0.6427`, delta `+0.0212`
- tick `72013`, seconds `12.50`, LSTM `0.6080`, delta `+0.0205`
- tick `71917`, seconds `11.00`, LSTM `0.6057`, delta `-0.0189`
- tick `72589`, seconds `21.50`, LSTM `0.6161`, delta `+0.0175`

## Top 15 local ridge features

- `lag_12__T_place_BALCONY`: coefficient `-0.001527`, |coef| `0.001527`
- `lag_14__T_place_BALCONY`: coefficient `0.001340`, |coef| `0.001340`
- `lag_00__CT_kills_last_3s`: coefficient `0.001308`, |coef| `0.001308`
- `lag_00__damage_diff_last_5s`: coefficient `0.001164`, |coef| `0.001164`
- `lag_00__CT_damage_last_5s`: coefficient `0.001149`, |coef| `0.001149`
- `lag_00__kill_diff_last_3s`: coefficient `0.001090`, |coef| `0.001090`
- `lag_00__T_place_BALCONY`: coefficient `-0.001002`, |coef| `0.001002`
- `lag_05__T_shots_fired_sum`: coefficient `-0.000909`, |coef| `0.000909`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000864`, |coef| `0.000864`
- `lag_07__T_shots_fired_sum`: coefficient `0.000863`, |coef| `0.000863`
- `lag_07__T3__shots_fired`: coefficient `0.000849`, |coef| `0.000849`
- `lag_00__T2__utility_total`: coefficient `-0.000809`, |coef| `0.000809`
- `lag_00__T3__shots_fired`: coefficient `0.000801`, |coef| `0.000801`
- `lag_13__T_place_SECONDMID`: coefficient `-0.000782`, |coef| `0.000782`
- `lag_10__CT_place_ARCH`: coefficient `0.000759`, |coef| `0.000759`

## Top 10 utility ridge features

- `lag_00__T2__utility_total`: coefficient `-0.000809` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000644` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000640` (lowers CT win probability)
- `lag_00__T2__smoke`: coefficient `-0.000635` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000601` (lowers CT win probability)
- `lag_14__T_A_site_active_infernos`: coefficient `-0.000540` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000537` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000533` (lowers CT win probability)
- `lag_15__T_A_site_active_infernos`: coefficient `-0.000522` (lowers CT win probability)
- `lag_00__T_utility_inv`: coefficient `-0.000520` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_12__T_place_BALCONY`: coefficient `-0.001527` (lowers CT win probability)
- `lag_14__T_place_BALCONY`: coefficient `0.001340` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001308` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001164` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001149` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001090` (raises CT win probability)
- `lag_00__T_place_BALCONY`: coefficient `-0.001002` (lowers CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `-0.000909` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000864` (raises CT win probability)
- `lag_07__T_shots_fired_sum`: coefficient `0.000863` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `73709`, seconds `39.00`, LSTM delta `+0.1302`

Top all feature movements:
- `lag_12__T_place_BALCONY`: contribution `+0.020993`
- `lag_13__T_place_BALCONY`: contribution `+0.008219`
- `lag_05__T_shots_fired_sum`: contribution `+0.006131`
- `lag_00__damage_diff_last_5s`: contribution `+0.004911`
- `lag_00__CT_damage_last_5s`: contribution `+0.004683`

Top utility-only movements:
- `lag_00__T2__utility_total`: contribution `+0.001990`

### tick `73741`, seconds `39.50`, LSTM delta `+0.1092`

Top all feature movements:
- `lag_14__T_place_BALCONY`: contribution `+0.018426`
- `lag_13__T_place_BALCONY`: contribution `-0.008219`
- `lag_00__CT_kills_last_3s`: contribution `+0.003776`
- `lag_06__T_shots_fired_sum`: contribution `+0.003606`
- `lag_07__T_shots_fired_sum`: contribution `+0.003235`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `+0.001554`

### tick `74669`, seconds `54.00`, LSTM delta `+0.0527`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.003776`
- `lag_00__damage_diff_last_5s`: contribution `+0.002626`
- `lag_00__kill_diff_last_3s`: contribution `+0.002625`
- `lag_00__CT_damage_last_5s`: contribution `+0.002504`
- `lag_00__CT5__is_scoped`: contribution `+0.002270`

Top utility-only movements:
- `lag_06__CT2__flash_duration`: contribution `+0.001925`
- `lag_06__CT_flash_duration_sum`: contribution `+0.001366`
- `lag_06__CT5__flash_duration`: contribution `+0.001350`

### tick `73325`, seconds `33.00`, LSTM delta `+0.0237`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `+0.013779`
- `lag_01__T_place_BALCONY`: contribution `+0.002048`
- `lag_03__T_place_LOWERMID`: contribution `+0.001842`
- `lag_00__T_place_APARTMENTS`: contribution `-0.001275`
- `lag_08__T2__duck_amount`: contribution `+0.001127`

Top utility-only movements:
- `lag_02__T_A_site_active_infernos`: contribution `+0.000555`

### tick `73293`, seconds `32.50`, LSTM delta `-0.0222`

Top all feature movements:
- `lag_00__T_place_BALCONY`: contribution `-0.013779`
- `lag_02__T_place_TRAMP`: contribution `-0.001689`
- `lag_15__T_A_site_active_infernos`: contribution `-0.001554`
- `lag_08__T2__duck_amount`: contribution `-0.001127`
- `lag_11__T3__is_walking`: contribution `+0.001041`

Top utility-only movements:
- `lag_15__T_A_site_active_infernos`: contribution `-0.001554`
- `lag_15__T_active_infernos`: contribution `-0.000794`
