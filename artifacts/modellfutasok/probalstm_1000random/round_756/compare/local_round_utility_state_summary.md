# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-m80-vs-rooster-bo3-GFAv4Fg83aXYKbsY0nLkP_/m80-vs-rooster-m2-inferno.csv`
- round_num: `7`
- rows: `138`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.783107 | 0.786696 | -0.003589 | 49 | 89 | 1.000000 | 1.000000 |
| active/recent utility | 138 | 1.000 | 0.783107 | 0.786696 | -0.003589 | 49 | 89 | 1.000000 | 1.000000 |
| strong utility action | 131 | 0.949 | 0.783907 | 0.787636 | -0.003729 | 45 | 86 | 1.000000 | 1.000000 |
| utility damage | 33 | 0.239 | 0.846708 | 0.851570 | -0.004862 | 11 | 22 | 1.000000 | 1.000000 |
| active smoke/inferno | 100 | 0.725 | 0.771119 | 0.765783 | 0.005336 | 44 | 56 | 1.000000 | 1.000000 |
| recent utility last 5s | 36 | 0.261 | 0.789134 | 0.799520 | -0.010386 | 11 | 25 | 1.000000 | 1.000000 |
| flash effect present | 138 | 1.000 | 0.783107 | 0.786696 | -0.003589 | 49 | 89 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `60.5s`, rows `100`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.7631`, XGBoost `0.6204`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7866`, XGBoost `0.6662`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7897`, XGBoost `0.6801`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.8449`, XGBoost `0.7693`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `71.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.7861`, XGBoost `0.7130`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.7870`, XGBoost `0.7149`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.6650`, XGBoost `0.7355`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `5`
- seconds `50.0`, LSTM `0.7828`, XGBoost `0.7130`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.8356`, XGBoost `0.7717`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `64.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.6684`, XGBoost `0.7314`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `5`
