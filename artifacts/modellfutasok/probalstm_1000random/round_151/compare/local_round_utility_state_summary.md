# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `4`
- rows: `215`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.399680 | 0.524293 | -0.124614 | 1 | 214 | 0.344186 | 0.809302 |
| active/recent utility | 215 | 1.000 | 0.399680 | 0.524293 | -0.124614 | 1 | 214 | 0.344186 | 0.809302 |
| strong utility action | 110 | 0.512 | 0.439442 | 0.538336 | -0.098894 | 0 | 110 | 0.390909 | 0.890909 |
| utility damage | 11 | 0.051 | 0.472808 | 0.570180 | -0.097372 | 0 | 11 | 0.363636 | 0.909091 |
| active smoke/inferno | 100 | 0.465 | 0.439371 | 0.536223 | -0.096852 | 0 | 100 | 0.430000 | 0.880000 |
| recent utility last 5s | 10 | 0.047 | 0.440156 | 0.559472 | -0.119316 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 215 | 1.000 | 0.399680 | 0.524293 | -0.124614 | 1 | 214 | 0.344186 | 0.809302 |

## Active Smoke/Inferno Intervals

- `8.5s` - `52.5s`, rows `89`
- `80.0s` - `85.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.0`, LSTM `0.3679`, XGBoost `0.5536`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.0258`, XGBoost `0.2103`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.4079`, XGBoost `0.5904`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.0282`, XGBoost `0.2085`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3833`, XGBoost `0.5611`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0315`, XGBoost `0.2085`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5616`, XGBoost `0.7381`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.0376`, XGBoost `0.2135`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.0384`, XGBoost `0.2135`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.4142`, XGBoost `0.5853`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
