# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `25`
- rows: `213`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 213 | 1.000 | 0.230056 | 0.226564 | 0.003492 | 98 | 115 | 0.234742 | 0.220657 |
| active/recent utility | 213 | 1.000 | 0.230056 | 0.226564 | 0.003492 | 98 | 115 | 0.234742 | 0.220657 |
| strong utility action | 184 | 0.864 | 0.223621 | 0.217815 | 0.005806 | 96 | 88 | 0.228261 | 0.184783 |
| utility damage | 20 | 0.094 | 0.342475 | 0.251824 | 0.090651 | 17 | 3 | 0.300000 | 0.050000 |
| active smoke/inferno | 184 | 0.864 | 0.223621 | 0.217815 | 0.005806 | 96 | 88 | 0.228261 | 0.184783 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 213 | 1.000 | 0.230056 | 0.226564 | 0.003492 | 98 | 115 | 0.234742 | 0.220657 |

## Active Smoke/Inferno Intervals

- `6.5s` - `58.0s`, rows `104`
- `66.5s` - `106.0s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `104.0`, LSTM `0.1819`, XGBoost `0.5688`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.1863`, XGBoost `0.5674`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.1804`, XGBoost `0.5609`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.1966`, XGBoost `0.5688`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.2138`, XGBoost `0.5688`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.3715`, XGBoost `0.6686`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5172`, XGBoost `0.3252`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `102.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.4726`, XGBoost `0.2838`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `10.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.0439`, XGBoost `0.2302`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5058`, XGBoost `0.3294`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `102.0`, recent_utility `0`
