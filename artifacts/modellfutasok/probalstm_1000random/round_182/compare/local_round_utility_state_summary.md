# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mibr-bo3-vjmAHfXA4PQfROTmirSCCF/vitality-vs-mibr-m2-inferno.csv`
- round_num: `7`
- rows: `212`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.196278 | 0.291268 | -0.094990 | 206 | 6 | 0.957547 | 0.943396 |
| active/recent utility | 212 | 1.000 | 0.196278 | 0.291268 | -0.094990 | 206 | 6 | 0.957547 | 0.943396 |
| strong utility action | 142 | 0.670 | 0.229674 | 0.319532 | -0.089858 | 138 | 4 | 0.936620 | 0.915493 |
| utility damage | 11 | 0.052 | 0.495556 | 0.545815 | -0.050259 | 11 | 0 | 0.363636 | 0.090909 |
| active smoke/inferno | 132 | 0.623 | 0.214444 | 0.308502 | -0.094058 | 128 | 4 | 0.931818 | 0.909091 |
| recent utility last 5s | 10 | 0.047 | 0.430718 | 0.465128 | -0.034410 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 212 | 1.000 | 0.196278 | 0.291268 | -0.094990 | 206 | 6 | 0.957547 | 0.943396 |

## Active Smoke/Inferno Intervals

- `7.0s` - `65.5s`, rows `118`
- `91.0s` - `97.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.0877`, XGBoost `0.3368`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0932`, XGBoost `0.3394`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.0962`, XGBoost `0.3348`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.1033`, XGBoost `0.3348`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.1174`, XGBoost `0.3368`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.1716`, XGBoost `0.3421`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.1670`, XGBoost `0.3368`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `61.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0928`, XGBoost `0.2599`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2140`, XGBoost `0.3740`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.1151`, XGBoost `0.2747`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
