# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `5`
- rows: `185`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 185 | 1.000 | 0.392291 | 0.467041 | -0.074749 | 28 | 157 | 0.372973 | 0.383784 |
| active/recent utility | 185 | 1.000 | 0.392291 | 0.467041 | -0.074749 | 28 | 157 | 0.372973 | 0.383784 |
| strong utility action | 169 | 0.914 | 0.372091 | 0.456617 | -0.084527 | 13 | 156 | 0.313609 | 0.325444 |
| utility damage | 10 | 0.054 | 0.255797 | 0.401133 | -0.145336 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 169 | 0.914 | 0.372091 | 0.456617 | -0.084527 | 13 | 156 | 0.313609 | 0.325444 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 185 | 1.000 | 0.392291 | 0.467041 | -0.074749 | 28 | 157 | 0.372973 | 0.383784 |

## Active Smoke/Inferno Intervals

- `8.0s` - `92.0s`, rows `169`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.1253`, XGBoost `0.4038`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.1561`, XGBoost `0.4000`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.1627`, XGBoost `0.4000`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3862`, XGBoost `0.6082`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.1389`, XGBoost `0.3602`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.1839`, XGBoost `0.4000`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.1445`, XGBoost `0.3602`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1649`, XGBoost `0.3602`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.2080`, XGBoost `0.4030`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.1672`, XGBoost `0.3602`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
