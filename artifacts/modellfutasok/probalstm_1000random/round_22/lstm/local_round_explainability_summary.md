# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `7`

## Largest probability jumps

- tick `56037`, seconds `41.00`, LSTM `0.8637`, delta `+0.2117`
- tick `56069`, seconds `41.50`, LSTM `0.7050`, delta `-0.1587`
- tick `55973`, seconds `40.00`, LSTM `0.6165`, delta `+0.1384`
- tick `54629`, seconds `19.00`, LSTM `0.4590`, delta `-0.1375`
- tick `55141`, seconds `27.00`, LSTM `0.5828`, delta `+0.0885`
- tick `55461`, seconds `32.00`, LSTM `0.4897`, delta `-0.0659`
- tick `56165`, seconds `43.00`, LSTM `0.7142`, delta `-0.0513`
- tick `56293`, seconds `45.00`, LSTM `0.7123`, delta `-0.0447`
- tick `56261`, seconds `44.50`, LSTM `0.7570`, delta `+0.0382`
- tick `56645`, seconds `50.50`, LSTM `0.7536`, delta `+0.0381`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003903`, |coef| `0.003903`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003165`, |coef| `0.003165`
- `lag_00__CT_kills_last_3s`: coefficient `0.003006`, |coef| `0.003006`
- `lag_00__damage_diff_last_5s`: coefficient `0.002920`, |coef| `0.002920`
- `lag_12__T2__duck_amount`: coefficient `0.002381`, |coef| `0.002381`
- `lag_08__T5__is_walking`: coefficient `0.001959`, |coef| `0.001959`
- `lag_02__CT_kills_last_3s`: coefficient `0.001939`, |coef| `0.001939`
- `lag_12__T_kills_last_3s`: coefficient `-0.001936`, |coef| `0.001936`
- `lag_05__CT3__duck_amount`: coefficient `-0.001874`, |coef| `0.001874`
- `lag_00__T_kills_last_3s`: coefficient `-0.001839`, |coef| `0.001839`
- `lag_02__T2__has_bomb`: coefficient `-0.001786`, |coef| `0.001786`
- `lag_02__kill_diff_last_3s`: coefficient `0.001698`, |coef| `0.001698`
- `lag_03__CT1__is_scoped`: coefficient `0.001684`, |coef| `0.001684`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001660`, |coef| `0.001660`
- `lag_00__T_macro_B`: coefficient `-0.001660`, |coef| `0.001660`

## Top 10 utility ridge features

- `lag_15__T_B_site_active_smokes`: coefficient `-0.001481` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `0.001407` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.001326` (raises CT win probability)
- `lag_15__T4__molly`: coefficient `-0.001188` (lowers CT win probability)
- `lag_06__T2__molly`: coefficient `-0.001170` (lowers CT win probability)
- `lag_15__T_active_smokes`: coefficient `-0.001117` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.001031` (raises CT win probability)
- `lag_02__T_A_site_active_smokes`: coefficient `-0.001027` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `-0.001025` (lowers CT win probability)
- `lag_04__T2__molly`: coefficient `-0.001011` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003903` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003165` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.003006` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002920` (raises CT win probability)
- `lag_12__T2__duck_amount`: coefficient `0.002381` (raises CT win probability)
- `lag_08__T5__is_walking`: coefficient `0.001959` (raises CT win probability)
- `lag_02__CT_kills_last_3s`: coefficient `0.001939` (raises CT win probability)
- `lag_12__T_kills_last_3s`: coefficient `-0.001936` (lowers CT win probability)
- `lag_05__CT3__duck_amount`: coefficient `-0.001874` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001839` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `56037`, seconds `41.00`, LSTM delta `+0.2117`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010993`
- `lag_00__kill_diff_last_3s`: contribution `+0.009395`
- `lag_00__CT_kills_last_3s`: contribution `+0.008679`
- `lag_03__CT1__is_scoped`: contribution `+0.007214`
- `lag_05__CT3__duck_amount`: contribution `+0.006975`

Top utility-only movements:
- `lag_11__T_B_site_active_infernos`: contribution `+0.003978`
- `lag_05__T_B_site_active_infernos`: contribution `+0.003750`

### tick `56069`, seconds `41.50`, LSTM delta `-0.1587`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.013191`
- `lag_00__kill_diff_last_3s`: contribution `-0.009395`
- `lag_03__CT1__is_scoped`: contribution `-0.007214`
- `lag_12__T2__duck_amount`: contribution `-0.006660`
- `lag_00__damage_diff_last_5s`: contribution `-0.006588`

Top utility-only movements:
- `lag_06__T_B_site_active_infernos`: contribution `-0.002898`
- `lag_15__T_B_site_active_smokes`: contribution `-0.002243`
- `lag_00__CT3__utility_total`: contribution `-0.001803`

### tick `55973`, seconds `40.00`, LSTM delta `+0.1384`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009395`
- `lag_00__CT_kills_last_3s`: contribution `+0.008679`
- `lag_15__T_shots_fired_sum`: contribution `+0.007101`
- `lag_00__damage_diff_last_5s`: contribution `+0.006588`
- `lag_14__CT_place_RUINS`: contribution `+0.005663`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `54629`, seconds `19.00`, LSTM delta `-0.1375`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009395`
- `lag_02__T4__flash_duration`: contribution `-0.006363`
- `lag_00__T_kills_last_3s`: contribution `-0.005826`
- `lag_00__damage_diff_last_5s`: contribution `-0.004809`
- `lag_08__T5__is_walking`: contribution `-0.004544`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `-0.006363`
- `lag_08__T5__flash_duration`: contribution `-0.003432`

### tick `55141`, seconds `27.00`, LSTM delta `+0.0885`

Top all feature movements:
- `lag_15__T_shots_fired_sum`: contribution `+0.009467`
- `lag_00__kill_diff_last_3s`: contribution `+0.009395`
- `lag_00__CT_kills_last_3s`: contribution `+0.008679`
- `lag_00__damage_diff_last_5s`: contribution `+0.006588`
- `lag_00__CT1__is_scoped`: contribution `+0.005324`

Top utility-only movements:
- `lag_09__CT_utility_damage_last_5s`: contribution `+0.001988`
