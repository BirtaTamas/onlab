# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `6`
- rows: `158`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.538667 | 0.610519 | -0.071852 | 153 | 5 | 0.386076 | 0.348101 |
| active/recent utility | 158 | 1.000 | 0.538667 | 0.610519 | -0.071852 | 153 | 5 | 0.386076 | 0.348101 |
| strong utility action | 103 | 0.652 | 0.555534 | 0.632230 | -0.076696 | 100 | 3 | 0.359223 | 0.300971 |
| utility damage | 10 | 0.063 | 0.042528 | 0.076968 | -0.034440 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 97 | 0.614 | 0.587094 | 0.666578 | -0.079485 | 94 | 3 | 0.319588 | 0.257732 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.538667 | 0.610519 | -0.071852 | 153 | 5 | 0.386076 | 0.348101 |

## Active Smoke/Inferno Intervals

- `12.5s` - `34.0s`, rows `44`
- `37.5s` - `59.5s`, rows `45`
- `75.0s` - `78.5s`, rows `8`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.2321`, XGBoost `0.0784`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.6470`, XGBoost `0.7919`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6369`, XGBoost `0.7753`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5997`, XGBoost `0.7367`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.6601`, XGBoost `0.7917`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.7164`, XGBoost `0.8409`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.4584`, XGBoost `0.5818`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0735`, XGBoost `0.1909`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.8279`, XGBoost `0.9446`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0779`, XGBoost `0.1942`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
