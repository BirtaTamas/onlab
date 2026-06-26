# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `15`
- rows: `187`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.734546 | 0.744602 | -0.010056 | 89 | 98 | 0.951872 | 0.882353 |
| active/recent utility | 187 | 1.000 | 0.734546 | 0.744602 | -0.010056 | 89 | 98 | 0.951872 | 0.882353 |
| strong utility action | 129 | 0.690 | 0.735886 | 0.733069 | 0.002817 | 64 | 65 | 0.984496 | 0.922481 |
| utility damage | 20 | 0.107 | 0.765234 | 0.772744 | -0.007509 | 10 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 129 | 0.690 | 0.735886 | 0.733069 | 0.002817 | 64 | 65 | 0.984496 | 0.922481 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 187 | 1.000 | 0.734546 | 0.744602 | -0.010056 | 89 | 98 | 0.951872 | 0.882353 |

## Active Smoke/Inferno Intervals

- `6.0s` - `70.0s`, rows `129`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.5`, LSTM `0.6195`, XGBoost `0.4793`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.6091`, XGBoost `0.7439`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.6057`, XGBoost `0.4807`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.5983`, XGBoost `0.4772`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.6256`, XGBoost `0.7331`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.5801`, XGBoost `0.4764`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.5611`, XGBoost `0.4587`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.7939`, XGBoost `0.8904`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `58.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.5774`, XGBoost `0.4873`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.6648`, XGBoost `0.5814`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
