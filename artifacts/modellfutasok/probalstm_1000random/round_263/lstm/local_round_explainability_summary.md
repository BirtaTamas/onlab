# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `5`

## Largest probability jumps

- tick `33825`, seconds `77.50`, LSTM `0.5582`, delta `+0.2727`
- tick `32161`, seconds `51.50`, LSTM `0.3062`, delta `-0.2643`
- tick `34849`, seconds `93.50`, LSTM `0.8794`, delta `+0.2621`
- tick `32769`, seconds `61.00`, LSTM `0.3251`, delta `+0.2228`
- tick `31361`, seconds `39.00`, LSTM `0.7522`, delta `+0.1920`
- tick `31649`, seconds `43.50`, LSTM `0.6002`, delta `-0.1796`
- tick `29633`, seconds `12.00`, LSTM `0.4898`, delta `+0.1592`
- tick `33761`, seconds `76.50`, LSTM `0.3223`, delta `-0.1238`
- tick `30305`, seconds `22.50`, LSTM `0.4140`, delta `-0.0886`
- tick `32833`, seconds `62.00`, LSTM `0.4574`, delta `+0.0778`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005528`, |coef| `0.005528`
- `lag_00__CT_kills_last_3s`: coefficient `0.004194`, |coef| `0.004194`
- `lag_14__CT_place_HUT`: coefficient `-0.003626`, |coef| `0.003626`
- `lag_00__T_shots_fired_sum`: coefficient `-0.003252`, |coef| `0.003252`
- `lag_10__CT1__duck_amount`: coefficient `0.003034`, |coef| `0.003034`
- `lag_01__CT_place_MINI`: coefficient `0.003004`, |coef| `0.003004`
- `lag_14__CT_place_MINI`: coefficient `-0.002884`, |coef| `0.002884`
- `lag_08__CT1__duck_amount`: coefficient `0.002809`, |coef| `0.002809`
- `lag_11__CT_place_HEAVEN`: coefficient `0.002743`, |coef| `0.002743`
- `lag_15__T2__duck_amount`: coefficient `0.002708`, |coef| `0.002708`
- `lag_00__T_kills_last_3s`: coefficient `-0.002674`, |coef| `0.002674`
- `lag_15__CT_place_HUT`: coefficient `-0.002667`, |coef| `0.002667`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002523`, |coef| `0.002523`
- `lag_02__CT_place_CRANE`: coefficient `-0.002515`, |coef| `0.002515`
- `lag_15__CT_place_VENTS`: coefficient `0.002510`, |coef| `0.002510`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002158` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.001300` (lowers CT win probability)
- `lag_00__T_mollies_last_5s`: coefficient `0.001295` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `-0.001195` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `-0.001127` (lowers CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.001089` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.001089` (lowers CT win probability)
- `lag_01__T_mollies_last_5s`: coefficient `0.001089` (raises CT win probability)
- `lag_00__T2__flash`: coefficient `-0.001047` (lowers CT win probability)
- `lag_05__T_flash_alpha_mean`: coefficient `-0.000997` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005528` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004194` (raises CT win probability)
- `lag_14__CT_place_HUT`: coefficient `-0.003626` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.003252` (lowers CT win probability)
- `lag_10__CT1__duck_amount`: coefficient `0.003034` (raises CT win probability)
- `lag_01__CT_place_MINI`: coefficient `0.003004` (raises CT win probability)
- `lag_14__CT_place_MINI`: coefficient `-0.002884` (lowers CT win probability)
- `lag_08__CT1__duck_amount`: coefficient `0.002809` (raises CT win probability)
- `lag_11__CT_place_HEAVEN`: coefficient `0.002743` (raises CT win probability)
- `lag_15__T2__duck_amount`: coefficient `0.002708` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `33825`, seconds `77.50`, LSTM delta `+0.2727`

Top all feature movements:
- `lag_02__CT_place_CRANE`: contribution `+0.041263`
- `lag_03__T_place_VENTS`: contribution `+0.031436`
- `lag_04__CT_place_SECRET`: contribution `+0.019456`
- `lag_09__CT_place_SECRET`: contribution `+0.018315`
- `lag_00__kill_diff_last_3s`: contribution `+0.013306`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `32161`, seconds `51.50`, LSTM delta `-0.2643`

Top all feature movements:
- `lag_14__CT_place_HUT`: contribution `-0.035362`
- `lag_00__CT_place_HUT`: contribution `-0.024138`
- `lag_11__CT_place_HEAVEN`: contribution `-0.014812`
- `lag_00__kill_diff_last_3s`: contribution `-0.013306`
- `lag_00__T_shots_fired_sum`: contribution `-0.012190`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `34849`, seconds `93.50`, LSTM delta `+0.2621`

Top all feature movements:
- `lag_07__CT_place_OBSERVATION`: contribution `+0.038101`
- `lag_10__T_velocity_mean`: contribution `+0.021673`
- `lag_15__CT_place_VENTS`: contribution `+0.021059`
- `lag_02__CT_place_OBSERVATION`: contribution `+0.017717`
- `lag_11__T_velocity_mean`: contribution `+0.017388`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.013091`
- `lag_00__T2__flash`: contribution `+0.003082`

### tick `32769`, seconds `61.00`, LSTM delta `+0.2228`

Top all feature movements:
- `lag_01__CT_place_SECRET`: contribution `+0.021908`
- `lag_08__CT_place_SECRET`: contribution `+0.018767`
- `lag_00__kill_diff_last_3s`: contribution `+0.013306`
- `lag_00__CT_kills_last_3s`: contribution `+0.012110`
- `lag_08__CT1__duck_amount`: contribution `+0.009488`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `31361`, seconds `39.00`, LSTM delta `+0.1920`

Top all feature movements:
- `lag_01__CT_place_MINI`: contribution `+0.018417`
- `lag_14__CT_place_MINI`: contribution `+0.017680`
- `lag_00__kill_diff_last_3s`: contribution `+0.013306`
- `lag_00__CT_kills_last_3s`: contribution `+0.012110`
- `lag_10__CT1__duck_amount`: contribution `+0.011578`

Top utility-only movements:
- No utility movement among the top local contributors.
