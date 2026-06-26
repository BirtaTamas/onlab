# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `53343`, seconds `73.50`, LSTM `0.3388`, delta `+0.2288`
- tick `53631`, seconds `78.00`, LSTM `0.1059`, delta `-0.2133`
- tick `50975`, seconds `36.50`, LSTM `0.2736`, delta `-0.1955`
- tick `53567`, seconds `77.00`, LSTM `0.3634`, delta `-0.1044`
- tick `52031`, seconds `53.00`, LSTM `0.0284`, delta `-0.1000`
- tick `51007`, seconds `37.00`, LSTM `0.1921`, delta `-0.0815`
- tick `50207`, seconds `24.50`, LSTM `0.5081`, delta `+0.0686`
- tick `49311`, seconds `10.50`, LSTM `0.4157`, delta `+0.0619`
- tick `53375`, seconds `74.00`, LSTM `0.4005`, delta `+0.0618`
- tick `53791`, seconds `80.50`, LSTM `0.0211`, delta `-0.0544`

## Top 15 local ridge features

- `lag_09__T_place_HELL`: coefficient `0.002553`, |coef| `0.002553`
- `lag_00__T_place_HELL`: coefficient `-0.002499`, |coef| `0.002499`
- `lag_00__kill_diff_last_3s`: coefficient `0.002116`, |coef| `0.002116`
- `lag_00__T_kills_last_3s`: coefficient `-0.001913`, |coef| `0.001913`
- `lag_02__T_place_HELL`: coefficient `-0.001628`, |coef| `0.001628`
- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001604`, |coef| `0.001604`
- `lag_13__T_place_HELL`: coefficient `0.001527`, |coef| `0.001527`
- `lag_00__damage_diff_last_5s`: coefficient `0.001468`, |coef| `0.001468`
- `lag_00__T_damage_last_5s`: coefficient `-0.001460`, |coef| `0.001460`
- `lag_06__CT3__duck_amount`: coefficient `-0.001435`, |coef| `0.001435`
- `lag_00__CT1__duck_amount`: coefficient `-0.001346`, |coef| `0.001346`
- `lag_01__kill_diff_last_3s`: coefficient `0.001339`, |coef| `0.001339`
- `lag_08__CT_place_VENTS`: coefficient `0.001314`, |coef| `0.001314`
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.001297`, |coef| `0.001297`
- `lag_06__CT_duck_amount_mean`: coefficient `-0.001253`, |coef| `0.001253`

## Top 10 utility ridge features

- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001604` (raises CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.001297` (raises CT win probability)
- `lag_14__CT2__flash_duration`: coefficient `0.000919` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.000817` (raises CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `0.000786` (raises CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `0.000641` (raises CT win probability)
- `lag_00__CT_B_site_active_smokes`: coefficient `0.000614` (raises CT win probability)
- `lag_00__CT_A_site_active_smokes`: coefficient `0.000611` (raises CT win probability)
- `lag_04__CT_A_site_active_smokes`: coefficient `0.000607` (raises CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.000575` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_HELL`: coefficient `0.002553` (raises CT win probability)
- `lag_00__T_place_HELL`: coefficient `-0.002499` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002116` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001913` (lowers CT win probability)
- `lag_02__T_place_HELL`: coefficient `-0.001628` (lowers CT win probability)
- `lag_13__T_place_HELL`: coefficient `0.001527` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001468` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001460` (lowers CT win probability)
- `lag_06__CT3__duck_amount`: coefficient `-0.001435` (lowers CT win probability)
- `lag_00__CT1__duck_amount`: coefficient `-0.001346` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `53343`, seconds `73.50`, LSTM delta `+0.2288`

Top all feature movements:
- `lag_09__T_place_HELL`: contribution `+0.054431`
- `lag_00__T_place_HELL`: contribution `+0.053293`
- `lag_13__T_place_HELL`: contribution `+0.032553`
- `lag_14__T_place_ADMIN`: contribution `+0.018333`
- `lag_07__T_place_HELL`: contribution `+0.014820`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `53631`, seconds `78.00`, LSTM delta `-0.2133`

Top all feature movements:
- `lag_09__T_place_HELL`: contribution `-0.054431`
- `lag_02__T_place_HELL`: contribution `-0.034711`
- `lag_00__T_place_CONTROL`: contribution `-0.006405`
- `lag_00__T_place_TROPHY`: contribution `-0.006291`
- `lag_00__T_kills_last_3s`: contribution `-0.006061`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `50975`, seconds `36.50`, LSTM delta `-0.1955`

Top all feature movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.017304`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.011476`
- `lag_08__CT_place_VENTS`: contribution `-0.011024`
- `lag_06__T_place_TROPHY`: contribution `-0.006077`
- `lag_00__T_kills_last_3s`: contribution `-0.006061`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.017304`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.011476`
- `lag_14__CT2__flash_duration`: contribution `-0.006036`
- `lag_15__T1__flash_duration`: contribution `-0.004459`

### tick `53567`, seconds `77.00`, LSTM delta `-0.1044`

Top all feature movements:
- `lag_00__T_place_HELL`: contribution `-0.053293`
- `lag_07__T_place_HELL`: contribution `+0.014820`
- `lag_14__T_place_HELL`: contribution `-0.010672`
- `lag_06__CT3__duck_amount`: contribution `+0.005339`
- `lag_05__T_place_TROPHY`: contribution `-0.005145`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `52031`, seconds `53.00`, LSTM delta `-0.1000`

Top all feature movements:
- `lag_15__CT_place_MINI`: contribution `-0.007315`
- `lag_00__T_kills_last_3s`: contribution `-0.006061`
- `lag_06__CT3__duck_amount`: contribution `-0.005339`
- `lag_00__kill_diff_last_3s`: contribution `-0.005094`
- `lag_00__CT1__duck_amount`: contribution `-0.004732`

Top utility-only movements:
- No utility movement among the top local contributors.
