# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-mouz-vs-falcons-bo3-ET1FlQ7LAGQtcSrRzzPcv6/mouz-vs-falcons-m1-dust2.csv`
- round_num: `12`
- rows: `125`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.757711 | 0.754082 | 0.003628 | 62 | 63 | 1.000000 | 1.000000 |
| active/recent utility | 125 | 1.000 | 0.757711 | 0.754082 | 0.003628 | 62 | 63 | 1.000000 | 1.000000 |
| strong utility action | 104 | 0.832 | 0.774220 | 0.769615 | 0.004605 | 51 | 53 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.080 | 0.907177 | 0.932825 | -0.025648 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 104 | 0.832 | 0.774220 | 0.769615 | 0.004605 | 51 | 53 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 125 | 1.000 | 0.757711 | 0.754082 | 0.003628 | 62 | 63 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `40.0s`, rows `62`
- `41.5s` - `62.0s`, rows `42`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.8902`, XGBoost `0.8145`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.8877`, XGBoost `0.8139`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.8872`, XGBoost `0.8143`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.7295`, XGBoost `0.6573`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.7199`, XGBoost `0.6573`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.8740`, XGBoost `0.8149`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.8755`, XGBoost `0.8192`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.8713`, XGBoost `0.8152`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.7086`, XGBoost `0.6573`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.8701`, XGBoost `0.8195`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
