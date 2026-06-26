# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-nemiga-vs-m80-bo3-A9YADMgFNfEy-U6IHDyx-U/nemiga-vs-m80-m2-dust2.csv`
- round_num: `8`
- rows: `136`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 136 | 1.000 | 0.192500 | 0.315230 | -0.122730 | 130 | 6 | 1.000000 | 0.830882 |
| active/recent utility | 136 | 1.000 | 0.192500 | 0.315230 | -0.122730 | 130 | 6 | 1.000000 | 0.830882 |
| strong utility action | 129 | 0.949 | 0.180273 | 0.305135 | -0.124862 | 123 | 6 | 1.000000 | 0.860465 |
| utility damage | 30 | 0.221 | 0.251595 | 0.335100 | -0.083505 | 26 | 4 | 1.000000 | 0.933333 |
| active smoke/inferno | 118 | 0.868 | 0.163212 | 0.287093 | -0.123881 | 112 | 6 | 1.000000 | 0.898305 |
| recent utility last 5s | 11 | 0.081 | 0.363294 | 0.498675 | -0.135380 | 11 | 0 | 1.000000 | 0.454545 |
| flash effect present | 136 | 1.000 | 0.192500 | 0.315230 | -0.122730 | 130 | 6 | 1.000000 | 0.830882 |

## Active Smoke/Inferno Intervals

- `9.0s` - `67.5s`, rows `118`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `62.0`, LSTM `0.0926`, XGBoost `0.3975`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0786`, XGBoost `0.3704`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0607`, XGBoost `0.3394`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0716`, XGBoost `0.3387`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0692`, XGBoost `0.3358`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0776`, XGBoost `0.3394`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0780`, XGBoost `0.3394`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0788`, XGBoost `0.3373`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1046`, XGBoost `0.3449`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1009`, XGBoost `0.3373`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
