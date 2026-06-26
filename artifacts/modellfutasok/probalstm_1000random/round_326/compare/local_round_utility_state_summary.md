# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `16`
- rows: `141`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 141 | 1.000 | 0.007355 | 0.019891 | -0.012535 | 98 | 43 | 1.000000 | 1.000000 |
| active/recent utility | 141 | 1.000 | 0.007355 | 0.019891 | -0.012535 | 98 | 43 | 1.000000 | 1.000000 |
| strong utility action | 90 | 0.638 | 0.008916 | 0.022563 | -0.013647 | 75 | 15 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 90 | 0.638 | 0.008916 | 0.022563 | -0.013647 | 75 | 15 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 141 | 1.000 | 0.007355 | 0.019891 | -0.012535 | 98 | 43 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `53.0s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.0068`, XGBoost `0.0445`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0081`, XGBoost `0.0442`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0075`, XGBoost `0.0435`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0073`, XGBoost `0.0430`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.0077`, XGBoost `0.0432`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0075`, XGBoost `0.0430`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.0112`, XGBoost `0.0445`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0119`, XGBoost `0.0451`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0051`, XGBoost `0.0373`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `23.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0054`, XGBoost `0.0373`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `23.0`, recent_utility `0`
