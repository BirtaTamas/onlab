# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `12`
- rows: `116`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.716564 | 0.702132 | 0.014432 | 71 | 45 | 0.965517 | 0.974138 |
| active/recent utility | 116 | 1.000 | 0.716564 | 0.702132 | 0.014432 | 71 | 45 | 0.965517 | 0.974138 |
| strong utility action | 113 | 0.974 | 0.716946 | 0.703978 | 0.012967 | 68 | 45 | 0.964602 | 0.973451 |
| utility damage | 20 | 0.172 | 0.632608 | 0.637912 | -0.005305 | 10 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 103 | 0.888 | 0.718052 | 0.710832 | 0.007220 | 58 | 45 | 0.961165 | 0.970874 |
| recent utility last 5s | 10 | 0.086 | 0.705550 | 0.633385 | 0.072164 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 116 | 1.000 | 0.716564 | 0.702132 | 0.014432 | 71 | 45 | 0.965517 | 0.974138 |

## Active Smoke/Inferno Intervals

- `6.5s` - `57.5s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.5837`, XGBoost `0.6724`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5876`, XGBoost `0.6724`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.7173`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.7154`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.7150`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.7110`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.7079`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.7061`, XGBoost `0.6334`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `14.0`, LSTM `0.6879`, XGBoost `0.6152`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6878`, XGBoost `0.6152`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
