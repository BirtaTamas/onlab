# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `6`

## Largest probability jumps

- tick `43706`, seconds `62.00`, LSTM `0.7513`, delta `+0.2366`
- tick `43354`, seconds `56.50`, LSTM `0.4284`, delta `+0.2231`
- tick `40730`, seconds `15.50`, LSTM `0.2104`, delta `-0.2025`
- tick `43770`, seconds `63.00`, LSTM `0.9397`, delta `+0.1903`
- tick `42554`, seconds `44.00`, LSTM `0.1918`, delta `+0.0754`
- tick `40762`, seconds `16.00`, LSTM `0.1524`, delta `-0.0580`
- tick `43386`, seconds `57.00`, LSTM `0.4863`, delta `+0.0579`
- tick `42810`, seconds `48.00`, LSTM `0.2165`, delta `-0.0526`
- tick `42682`, seconds `46.00`, LSTM `0.2721`, delta `+0.0517`
- tick `40794`, seconds `16.50`, LSTM `0.1028`, delta `-0.0496`

## Top 15 local ridge features

- `lag_05__CT_place_TUNNEL`: coefficient `0.002683`, |coef| `0.002683`
- `lag_09__CT_place_TUNNEL`: coefficient `-0.002492`, |coef| `0.002492`
- `lag_01__CT_place_OUTSIDELONG`: coefficient `-0.002381`, |coef| `0.002381`
- `lag_14__CT_place_BRIDGE`: coefficient `0.002161`, |coef| `0.002161`
- `lag_04__CT_place_BRIDGE`: coefficient `-0.002113`, |coef| `0.002113`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001883`, |coef| `0.001883`
- `lag_01__T1__is_scoped`: coefficient `0.001776`, |coef| `0.001776`
- `lag_00__damage_diff_last_5s`: coefficient `0.001765`, |coef| `0.001765`
- `lag_00__kill_diff_last_3s`: coefficient `0.001717`, |coef| `0.001717`
- `lag_03__CT_place_FOUNTAIN`: coefficient `-0.001697`, |coef| `0.001697`
- `lag_06__CT_place_BRIDGE`: coefficient `-0.001676`, |coef| `0.001676`
- `lag_00__CT_kills_last_3s`: coefficient `0.001590`, |coef| `0.001590`
- `lag_09__CT_place_MAIN`: coefficient `-0.001590`, |coef| `0.001590`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001431`, |coef| `0.001431`
- `lag_00__CT_damage_last_5s`: coefficient `0.001345`, |coef| `0.001345`

## Top 10 utility ridge features

- `lag_01__T_B_site_active_infernos`: coefficient `0.001234` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `0.000951` (raises CT win probability)
- `lag_01__T1__flash_duration`: coefficient `-0.000874` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `-0.000863` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.000848` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.000781` (lowers CT win probability)
- `lag_01__active_infernos_total`: coefficient `0.000750` (raises CT win probability)
- `lag_12__CT1__molly`: coefficient `-0.000710` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000646` (lowers CT win probability)
- `lag_12__T_active_infernos`: coefficient `0.000622` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_place_TUNNEL`: coefficient `0.002683` (raises CT win probability)
- `lag_09__CT_place_TUNNEL`: coefficient `-0.002492` (lowers CT win probability)
- `lag_01__CT_place_OUTSIDELONG`: coefficient `-0.002381` (lowers CT win probability)
- `lag_14__CT_place_BRIDGE`: coefficient `0.002161` (raises CT win probability)
- `lag_04__CT_place_BRIDGE`: coefficient `-0.002113` (lowers CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001883` (raises CT win probability)
- `lag_01__T1__is_scoped`: coefficient `0.001776` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001765` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001717` (raises CT win probability)
- `lag_03__CT_place_FOUNTAIN`: coefficient `-0.001697` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `43706`, seconds `62.00`, LSTM delta `+0.2366`

Top all feature movements:
- `lag_09__CT_place_TUNNEL`: contribution `+0.040031`
- `lag_04__CT_place_BRIDGE`: contribution `+0.024219`
- `lag_09__CT_place_MAIN`: contribution `+0.010704`
- `lag_01__T1__is_scoped`: contribution `+0.010148`
- `lag_07__CT_place_MAIN`: contribution `+0.007057`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `+0.003488`
- `lag_12__T_B_site_active_infernos`: contribution `+0.002398`

### tick `43354`, seconds `56.50`, LSTM delta `+0.2231`

Top all feature movements:
- `lag_05__CT_place_TUNNEL`: contribution `+0.043099`
- `lag_14__CT_place_BRIDGE`: contribution `+0.024773`
- `lag_03__CT_place_FOUNTAIN`: contribution `+0.017850`
- `lag_08__CT_place_FOUNTAIN`: contribution `+0.007534`
- `lag_00__CT_kills_last_3s`: contribution `+0.004591`

Top utility-only movements:
- `lag_01__T_B_site_active_infernos`: contribution `+0.003488`

### tick `40730`, seconds `15.50`, LSTM delta `-0.2025`

Top all feature movements:
- `lag_01__CT_place_OUTSIDELONG`: contribution `-0.048305`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.019096`
- `lag_10__T_shots_fired_sum`: contribution `-0.006318`
- `lag_10__T_place_STREET`: contribution `-0.006312`
- `lag_00__T1__is_scoped`: contribution `-0.005980`

Top utility-only movements:
- `lag_01__T1__flash_duration`: contribution `-0.003884`

### tick `43770`, seconds `63.00`, LSTM delta `+0.1903`

Top all feature movements:
- `lag_06__CT_place_BRIDGE`: contribution `+0.019208`
- `lag_11__CT_place_TUNNEL`: contribution `+0.017364`
- `lag_09__CT_place_MAIN`: contribution `+0.010704`
- `lag_00__CT_place_CTSIDEUPPER`: contribution `+0.007387`
- `lag_10__CT_place_MAIN`: contribution `+0.005559`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `42554`, seconds `44.00`, LSTM delta `+0.0754`

Top all feature movements:
- `lag_06__CT_place_FOUNTAIN`: contribution `+0.008056`
- `lag_00__T1__is_scoped`: contribution `-0.005980`
- `lag_00__CT_kills_last_3s`: contribution `+0.004591`
- `lag_00__kill_diff_last_3s`: contribution `+0.004132`
- `lag_04__T_shots_fired_sum`: contribution `+0.003898`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.002690`
