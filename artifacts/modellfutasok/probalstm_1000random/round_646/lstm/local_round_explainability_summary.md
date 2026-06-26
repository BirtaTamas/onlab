# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `7`

## Largest probability jumps

- tick `57238`, seconds `99.00`, LSTM `0.7177`, delta `+0.2914`
- tick `54166`, seconds `51.00`, LSTM `0.1557`, delta `-0.2475`
- tick `57110`, seconds `97.00`, LSTM `0.2834`, delta `+0.2205`
- tick `55158`, seconds `66.50`, LSTM `0.1989`, delta `+0.1304`
- tick `55830`, seconds `77.00`, LSTM `0.4823`, delta `+0.1292`
- tick `56854`, seconds `93.00`, LSTM `0.1228`, delta `-0.1175`
- tick `57206`, seconds `98.50`, LSTM `0.4263`, delta `+0.1102`
- tick `56086`, seconds `81.00`, LSTM `0.3420`, delta `-0.0914`
- tick `55702`, seconds `75.00`, LSTM `0.5046`, delta `+0.0901`
- tick `56342`, seconds `85.00`, LSTM `0.2792`, delta `-0.0895`

## Top 15 local ridge features

- `lag_00__T_place_DECON`: coefficient `-0.003519`, |coef| `0.003519`
- `lag_00__kill_diff_last_3s`: coefficient `0.002766`, |coef| `0.002766`
- `lag_10__T_shots_fired_sum`: coefficient `-0.002747`, |coef| `0.002747`
- `lag_00__T_place_VENTS`: coefficient `-0.002714`, |coef| `0.002714`
- `lag_00__T1__is_scoped`: coefficient `0.002651`, |coef| `0.002651`
- `lag_12__T1__is_scoped`: coefficient `-0.002498`, |coef| `0.002498`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002367`, |coef| `0.002367`
- `lag_10__T4__shots_fired`: coefficient `-0.002289`, |coef| `0.002289`
- `lag_14__T4__shots_fired`: coefficient `-0.002274`, |coef| `0.002274`
- `lag_11__T1__is_scoped`: coefficient `-0.002181`, |coef| `0.002181`
- `lag_14__T_shots_fired_sum`: coefficient `-0.002118`, |coef| `0.002118`
- `lag_04__CT_place_DECON`: coefficient `-0.002099`, |coef| `0.002099`
- `lag_02__T4__shots_fired`: coefficient `0.001948`, |coef| `0.001948`
- `lag_04__T_place_DECON`: coefficient `-0.001923`, |coef| `0.001923`
- `lag_00__T_kills_last_3s`: coefficient `-0.001915`, |coef| `0.001915`

## Top 10 utility ridge features

- `lag_03__T_A_site_active_infernos`: coefficient `-0.001011` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `-0.000964` (lowers CT win probability)
- `lag_08__CT_A_site_active_smokes`: coefficient `-0.000821` (lowers CT win probability)
- `lag_08__CT_B_site_active_smokes`: coefficient `-0.000800` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.000730` (lowers CT win probability)
- `lag_01__CT_A_site_active_infernos`: coefficient `0.000703` (raises CT win probability)
- `lag_01__CT_B_site_active_infernos`: coefficient `0.000682` (raises CT win probability)
- `lag_03__T5__molly`: coefficient `-0.000672` (lowers CT win probability)
- `lag_08__CT_A_site_active_infernos`: coefficient `0.000671` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000670` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_DECON`: coefficient `-0.003519` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002766` (raises CT win probability)
- `lag_10__T_shots_fired_sum`: coefficient `-0.002747` (lowers CT win probability)
- `lag_00__T_place_VENTS`: coefficient `-0.002714` (lowers CT win probability)
- `lag_00__T1__is_scoped`: coefficient `0.002651` (raises CT win probability)
- `lag_12__T1__is_scoped`: coefficient `-0.002498` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002367` (raises CT win probability)
- `lag_10__T4__shots_fired`: coefficient `-0.002289` (lowers CT win probability)
- `lag_14__T4__shots_fired`: coefficient `-0.002274` (lowers CT win probability)
- `lag_11__T1__is_scoped`: coefficient `-0.002181` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `57238`, seconds `99.00`, LSTM delta `+0.2914`

Top all feature movements:
- `lag_04__T_place_DECON`: contribution `+0.030892`
- `lag_14__T_shots_fired_sum`: contribution `+0.027001`
- `lag_14__T4__shots_fired`: contribution `+0.023878`
- `lag_12__T1__is_scoped`: contribution `+0.014271`
- `lag_02__T1__is_scoped`: contribution `+0.010029`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `54166`, seconds `51.00`, LSTM delta `-0.2475`

Top all feature movements:
- `lag_13__CT_place_VENDING`: contribution `-0.029985`
- `lag_08__CT_place_VENDING`: contribution `-0.022625`
- `lag_15__CT_place_TROPHY`: contribution `-0.022338`
- `lag_12__CT_place_TROPHY`: contribution `-0.016598`
- `lag_04__CT_place_TROPHY`: contribution `-0.015212`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `57110`, seconds `97.00`, LSTM delta `+0.2205`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `+0.056531`
- `lag_10__T_shots_fired_sum`: contribution `+0.035014`
- `lag_10__T4__shots_fired`: contribution `+0.024038`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008222`
- `lag_00__kill_diff_last_3s`: contribution `+0.006657`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55158`, seconds `66.50`, LSTM delta `+0.1304`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `+0.056531`
- `lag_08__T_place_DECON`: contribution `-0.015528`
- `lag_05__T_place_DECON`: contribution `+0.014886`
- `lag_12__T1__is_scoped`: contribution `-0.014271`
- `lag_11__T1__is_scoped`: contribution `+0.012461`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55830`, seconds `77.00`, LSTM delta `+0.1292`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `+0.056531`
- `lag_04__T_place_DECON`: contribution `+0.030892`
- `lag_15__T_place_OBSERVATION`: contribution `+0.014871`
- `lag_07__T4__duck_amount`: contribution `+0.006020`
- `lag_04__T5__is_walking`: contribution `+0.004360`

Top utility-only movements:
- `lag_08__CT_B_site_active_smokes`: contribution `-0.001328`
