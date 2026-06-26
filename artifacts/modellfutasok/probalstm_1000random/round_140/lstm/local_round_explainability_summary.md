# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `3`

## Largest probability jumps

- tick `15361`, seconds `48.00`, LSTM `0.6140`, delta `-0.2283`
- tick `14977`, seconds `42.00`, LSTM `0.5597`, delta `+0.1614`
- tick `15041`, seconds `43.00`, LSTM `0.7601`, delta `+0.1429`
- tick `14881`, seconds `40.50`, LSTM `0.4411`, delta `-0.1280`
- tick `15137`, seconds `44.50`, LSTM `0.8918`, delta `+0.1141`
- tick `15489`, seconds `50.00`, LSTM `0.5303`, delta `-0.0927`
- tick `15553`, seconds `51.00`, LSTM `0.6135`, delta `+0.0927`
- tick `14913`, seconds `41.00`, LSTM `0.3549`, delta `-0.0862`
- tick `15329`, seconds `47.50`, LSTM `0.8423`, delta `-0.0607`
- tick `15009`, seconds `42.50`, LSTM `0.6172`, delta `+0.0575`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001882`, |coef| `0.001882`
- `lag_00__T4__flash_duration`: coefficient `0.001372`, |coef| `0.001372`
- `lag_00__T_kills_last_3s`: coefficient `-0.001271`, |coef| `0.001271`
- `lag_00__CT_place_LONGDOG`: coefficient `0.001223`, |coef| `0.001223`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001195`, |coef| `0.001195`
- `lag_00__damage_diff_last_5s`: coefficient `0.001176`, |coef| `0.001176`
- `lag_00__CT_kills_last_3s`: coefficient `0.001099`, |coef| `0.001099`
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000979`, |coef| `0.000979`
- `lag_02__bomb_events_last_5s`: coefficient `0.000977`, |coef| `0.000977`
- `lag_06__T_B_site_active_infernos`: coefficient `0.000962`, |coef| `0.000962`
- `lag_05__T_B_site_active_infernos`: coefficient `0.000946`, |coef| `0.000946`
- `lag_01__kill_diff_last_3s`: coefficient `0.000918`, |coef| `0.000918`
- `lag_01__CT1__flash_duration`: coefficient `0.000911`, |coef| `0.000911`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000902`, |coef| `0.000902`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000896`, |coef| `0.000896`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `0.001372` (raises CT win probability)
- `lag_00__T_B_site_active_infernos`: coefficient `-0.000979` (lowers CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.000962` (raises CT win probability)
- `lag_05__T_B_site_active_infernos`: coefficient `0.000946` (raises CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000911` (raises CT win probability)
- `lag_01__T_B_site_active_infernos`: coefficient `-0.000869` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.000855` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `0.000749` (raises CT win probability)
- `lag_00__T_active_infernos`: coefficient `-0.000721` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.000717` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001882` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001271` (lowers CT win probability)
- `lag_00__CT_place_LONGDOG`: coefficient `0.001223` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001195` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001176` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001099` (raises CT win probability)
- `lag_02__bomb_events_last_5s`: coefficient `0.000977` (raises CT win probability)
- `lag_01__kill_diff_last_3s`: coefficient `0.000918` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.000902` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000896` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `15361`, seconds `48.00`, LSTM delta `-0.2283`

Top all feature movements:
- `lag_00__T4__flash_duration`: contribution `-0.008908`
- `lag_01__T_shots_fired_sum`: contribution `-0.005373`
- `lag_15__CT_place_LONGDOG`: contribution `-0.005196`
- `lag_00__kill_diff_last_3s`: contribution `-0.004530`
- `lag_11__T4__flash_duration`: contribution `-0.004356`

Top utility-only movements:
- `lag_00__T4__flash_duration`: contribution `-0.008908`
- `lag_11__T4__flash_duration`: contribution `-0.004356`
- `lag_02__CT1__flash_duration`: contribution `-0.003875`
- `lag_08__T1__flash_duration`: contribution `-0.003430`
- `lag_11__CT1__flash_duration`: contribution `-0.002627`

### tick `14977`, seconds `42.00`, LSTM delta `+0.1614`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.004530`
- `lag_03__CT_place_LONGDOG`: contribution `+0.004448`
- `lag_00__CT_shots_fired_sum`: contribution `+0.004386`
- `lag_08__CT2__duck_amount`: contribution `+0.003314`
- `lag_00__bomb_events_last_5s`: contribution `+0.003231`

Top utility-only movements:
- `lag_06__T_B_site_active_infernos`: contribution `+0.002720`
- `lag_11__CT_A_site_active_infernos`: contribution `+0.002045`

### tick `15041`, seconds `43.00`, LSTM delta `+0.1429`

Top all feature movements:
- `lag_01__T4__flash_duration`: contribution `+0.005551`
- `lag_01__CT1__flash_duration`: contribution `+0.004711`
- `lag_00__T_place_BACKOFB`: contribution `+0.004693`
- `lag_00__kill_diff_last_3s`: contribution `+0.004530`
- `lag_05__CT_place_LONGDOG`: contribution `+0.004367`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.005551`
- `lag_01__CT1__flash_duration`: contribution `+0.004711`
- `lag_06__T_B_site_active_infernos`: contribution `+0.002720`
- `lag_05__T_B_site_active_infernos`: contribution `+0.002675`
- `lag_01__T_B_site_active_infernos`: contribution `+0.002458`

### tick `14881`, seconds `40.50`, LSTM delta `-0.1280`

Top all feature movements:
- `lag_00__CT_place_LONGDOG`: contribution `-0.007975`
- `lag_00__kill_diff_last_3s`: contribution `-0.004530`
- `lag_00__T_kills_last_3s`: contribution `-0.004028`
- `lag_00__T_shots_fired_sum`: contribution `-0.003358`
- `lag_08__CT2__duck_amount`: contribution `-0.003314`

Top utility-only movements:
- `lag_00__T_B_site_active_infernos`: contribution `-0.002768`
- `lag_01__T_B_site_active_infernos`: contribution `-0.002458`
- `lag_08__CT_A_site_active_infernos`: contribution `-0.002342`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.002251`
- `lag_13__CT_B_site_active_infernos`: contribution `-0.002139`

### tick `15137`, seconds `44.50`, LSTM delta `+0.1141`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.004530`
- `lag_04__T4__flash_duration`: contribution `+0.004168`
- `lag_12__CT2__duck_amount`: contribution `+0.003411`
- `lag_01__T1__flash_duration`: contribution `+0.003365`
- `lag_00__CT_kills_last_3s`: contribution `+0.003173`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.004168`
- `lag_01__T1__flash_duration`: contribution `+0.003365`
- `lag_04__CT1__flash_duration`: contribution `+0.002158`
