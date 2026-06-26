# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `17197`, seconds `0.50`, LSTM `0.9521`, delta `+0.0273`
- tick `18765`, seconds `25.00`, LSTM `0.9310`, delta `+0.0186`
- tick `19853`, seconds `42.00`, LSTM `0.9204`, delta `-0.0176`
- tick `20301`, seconds `49.00`, LSTM `0.9505`, delta `+0.0173`
- tick `21165`, seconds `62.50`, LSTM `0.9330`, delta `-0.0172`
- tick `19597`, seconds `38.00`, LSTM `0.9176`, delta `-0.0165`
- tick `18893`, seconds `27.00`, LSTM `0.9103`, delta `-0.0155`
- tick `19725`, seconds `40.00`, LSTM `0.9404`, delta `+0.0154`
- tick `22029`, seconds `76.00`, LSTM `0.9363`, delta `+0.0152`
- tick `22541`, seconds `84.00`, LSTM `0.9601`, delta `+0.0146`

## Top 15 local ridge features

- `lag_00__CT_walking_count`: coefficient `-0.000636`, |coef| `0.000636`
- `lag_00__CT5__is_walking`: coefficient `-0.000566`, |coef| `0.000566`
- `lag_00__CT2__duck_amount`: coefficient `0.000543`, |coef| `0.000543`
- `lag_00__CT_smokes_last_5s`: coefficient `0.000428`, |coef| `0.000428`
- `lag_00__CT2__is_walking`: coefficient `-0.000401`, |coef| `0.000401`
- `lag_00__CT3__is_walking`: coefficient `-0.000356`, |coef| `0.000356`
- `lag_00__T_walking_count`: coefficient `-0.000356`, |coef| `0.000356`
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.000351`, |coef| `0.000351`
- `lag_10__T_place_SNIPERSNEST`: coefficient `-0.000326`, |coef| `0.000326`
- `lag_00__T_place_HOUSE`: coefficient `-0.000324`, |coef| `0.000324`
- `lag_10__CT_place_JUNGLE`: coefficient `0.000312`, |coef| `0.000312`
- `lag_15__T_place_HOUSE`: coefficient `0.000304`, |coef| `0.000304`
- `lag_00__T_place_SIDEALLEY`: coefficient `0.000282`, |coef| `0.000282`
- `lag_00__T4__is_walking`: coefficient `-0.000280`, |coef| `0.000280`
- `lag_06__CT_place_JUNGLE`: coefficient `-0.000278`, |coef| `0.000278`

## Top 10 utility ridge features

- `lag_00__CT_smokes_last_5s`: coefficient `0.000428` (raises CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `-0.000258` (lowers CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `-0.000253` (lowers CT win probability)
- `lag_00__CT_active_smokes`: coefficient `-0.000217` (lowers CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `-0.000205` (lowers CT win probability)
- `lag_10__CT_smokes_last_5s`: coefficient `0.000176` (raises CT win probability)
- `lag_04__CT_active_smokes`: coefficient `-0.000172` (lowers CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `-0.000168` (lowers CT win probability)
- `lag_05__CT_utility_damage_last_5s`: coefficient `-0.000167` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000151` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_walking_count`: coefficient `-0.000636` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000566` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.000543` (raises CT win probability)
- `lag_00__CT2__is_walking`: coefficient `-0.000401` (lowers CT win probability)
- `lag_00__CT3__is_walking`: coefficient `-0.000356` (lowers CT win probability)
- `lag_00__T_walking_count`: coefficient `-0.000356` (lowers CT win probability)
- `lag_00__T_place_SNIPERSNEST`: coefficient `-0.000351` (lowers CT win probability)
- `lag_10__T_place_SNIPERSNEST`: coefficient `-0.000326` (lowers CT win probability)
- `lag_00__T_place_HOUSE`: coefficient `-0.000324` (lowers CT win probability)
- `lag_10__CT_place_JUNGLE`: coefficient `0.000312` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `17197`, seconds `0.50`, LSTM delta `+0.0273`

Top all feature movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.007405`
- `lag_01__CT_place_CTSPAWN`: contribution `+0.000923`
- `lag_00__CT_velocity_mean`: contribution `+0.000622`
- `lag_00__T_velocity_mean`: contribution `+0.000583`
- `lag_01__T_place_TSPAWN`: contribution `+0.000489`

Top utility-only movements:
- `lag_00__CT_smokes_last_5s`: contribution `+0.007405`
- `lag_01__utility_inv_diff`: contribution `+0.000234`
- `lag_01__molly_inv_diff`: contribution `+0.000221`
- `lag_00__T4__smoke`: contribution `+0.000193`
- `lag_00__CT1__smoke`: contribution `+0.000189`

### tick `18765`, seconds `25.00`, LSTM delta `+0.0186`

Top all feature movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.001949`
- `lag_06__CT_place_JUNGLE`: contribution `+0.001786`
- `lag_00__CT_walking_count`: contribution `+0.001712`
- `lag_00__CT5__is_walking`: contribution `+0.001357`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.001307`

Top utility-only movements:
- `lag_05__CT_utility_damage_last_5s`: contribution `+0.001949`
- `lag_05__utility_damage_diff_last_5s`: contribution `+0.001307`
- `lag_15__CT_utility_damage_last_5s`: contribution `+0.000753`
- `lag_15__utility_damage_diff_last_5s`: contribution `+0.000520`

### tick `19853`, seconds `42.00`, LSTM delta `-0.0176`

Top all feature movements:
- `lag_10__CT_place_JUNGLE`: contribution `-0.002002`
- `lag_00__CT5__is_walking`: contribution `-0.001357`
- `lag_15__T_place_HOUSE`: contribution `-0.001335`
- `lag_00__CT_walking_count`: contribution `-0.001142`
- `lag_04__CT2__duck_amount`: contribution `-0.000599`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20301`, seconds `49.00`, LSTM delta `+0.0173`

Top all feature movements:
- `lag_00__CT2__duck_amount`: contribution `+0.002070`
- `lag_00__CT5__is_walking`: contribution `+0.001357`
- `lag_13__CT_place_CATWALK`: contribution `+0.000755`
- `lag_13__CT_place_CONNECTOR`: contribution `+0.000719`
- `lag_00__T3__is_walking`: contribution `+0.000612`

Top utility-only movements:
- `lag_00__CT_B_site_active_smokes`: contribution `+0.000429`
- `lag_00__CT_A_site_active_smokes`: contribution `+0.000407`

### tick `21165`, seconds `62.50`, LSTM delta `-0.0172`

Top all feature movements:
- `lag_00__CT2__duck_amount`: contribution `-0.002070`
- `lag_00__CT_walking_count`: contribution `-0.001712`
- `lag_14__CT_place_JUNGLE`: contribution `-0.001128`
- `lag_00__CT2__is_walking`: contribution `-0.000947`
- `lag_00__CT3__is_walking`: contribution `-0.000851`

Top utility-only movements:
- No utility movement among the top local contributors.
