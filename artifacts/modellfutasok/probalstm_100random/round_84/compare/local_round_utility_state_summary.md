# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `1`
- rows: `161`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 161 | 1.000 | 0.290914 | 0.336467 | -0.045552 | 161 | 0 | 0.496894 | 0.453416 |
| active/recent utility | 161 | 1.000 | 0.290914 | 0.336467 | -0.045552 | 161 | 0 | 0.496894 | 0.453416 |
| strong utility action | 88 | 0.547 | 0.308606 | 0.362616 | -0.054009 | 88 | 0 | 0.500000 | 0.431818 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.547 | 0.308606 | 0.362616 | -0.054009 | 88 | 0 | 0.500000 | 0.431818 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 161 | 1.000 | 0.290914 | 0.336467 | -0.045552 | 161 | 0 | 0.496894 | 0.453416 |

## Active Smoke/Inferno Intervals

- `14.0s` - `35.5s`, rows `44`
- `41.0s` - `62.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `46.5`, LSTM `0.0355`, XGBoost `0.2477`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0389`, XGBoost `0.2477`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0608`, XGBoost `0.2477`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.0669`, XGBoost `0.2477`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0495`, XGBoost `0.2225`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0561`, XGBoost `0.1799`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0662`, XGBoost `0.1831`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.4900`, XGBoost `0.5982`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.4859`, XGBoost `0.5932`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.4637`, XGBoost `0.5613`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
