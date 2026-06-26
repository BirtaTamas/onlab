# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `12`
- rows: `194`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.087953 | 0.113527 | -0.025574 | 192 | 2 | 0.979381 | 0.855670 |
| active/recent utility | 194 | 1.000 | 0.087953 | 0.113527 | -0.025574 | 192 | 2 | 0.979381 | 0.855670 |
| strong utility action | 153 | 0.789 | 0.048100 | 0.067841 | -0.019741 | 151 | 2 | 0.986928 | 0.947712 |
| utility damage | 20 | 0.103 | 0.186193 | 0.231780 | -0.045587 | 18 | 2 | 1.000000 | 0.750000 |
| active smoke/inferno | 153 | 0.789 | 0.048100 | 0.067841 | -0.019741 | 151 | 2 | 0.986928 | 0.947712 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 194 | 1.000 | 0.087953 | 0.113527 | -0.025574 | 192 | 2 | 0.979381 | 0.855670 |

## Active Smoke/Inferno Intervals

- `10.0s` - `38.0s`, rows `57`
- `42.0s` - `63.5s`, rows `44`
- `71.0s` - `96.5s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.2533`, XGBoost `0.3722`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `109.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4268`, XGBoost `0.5175`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `109.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0588`, XGBoost `0.1433`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `57.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.4452`, XGBoost `0.5225`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `109.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4374`, XGBoost `0.5071`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `109.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0496`, XGBoost `0.1193`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `33.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0492`, XGBoost `0.1184`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `50.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0763`, XGBoost `0.1433`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0632`, XGBoost `0.1166`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `109.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0847`, XGBoost `0.1373`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `24.0`, recent_utility `0`
