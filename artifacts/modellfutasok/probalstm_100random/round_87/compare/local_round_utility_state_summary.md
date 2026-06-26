# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `4`
- rows: `225`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 225 | 1.000 | 0.360765 | 0.439024 | -0.078259 | 195 | 30 | 0.564444 | 0.475556 |
| active/recent utility | 225 | 1.000 | 0.360765 | 0.439024 | -0.078259 | 195 | 30 | 0.564444 | 0.475556 |
| strong utility action | 159 | 0.707 | 0.453175 | 0.527408 | -0.074233 | 129 | 30 | 0.465409 | 0.345912 |
| utility damage | 17 | 0.076 | 0.373014 | 0.434826 | -0.061813 | 16 | 1 | 0.588235 | 0.588235 |
| active smoke/inferno | 159 | 0.707 | 0.453175 | 0.527408 | -0.074233 | 129 | 30 | 0.465409 | 0.345912 |
| recent utility last 5s | 27 | 0.120 | 0.439327 | 0.520724 | -0.081397 | 26 | 1 | 0.333333 | 0.296296 |
| flash effect present | 225 | 1.000 | 0.360765 | 0.439024 | -0.078259 | 195 | 30 | 0.564444 | 0.475556 |

## Active Smoke/Inferno Intervals

- `6.5s` - `85.5s`, rows `159`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.2741`, XGBoost `0.5240`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.2872`, XGBoost `0.5240`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.2998`, XGBoost `0.5273`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6099`, XGBoost `0.8353`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.3096`, XGBoost `0.5265`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6193`, XGBoost `0.8361`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6191`, XGBoost `0.8355`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6229`, XGBoost `0.8355`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.3165`, XGBoost `0.5265`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6283`, XGBoost `0.8346`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
