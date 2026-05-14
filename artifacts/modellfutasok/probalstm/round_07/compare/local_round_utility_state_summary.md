# Local Round Utility State Analysis

- csv_path: `processed_full\blast_austin_major_stage_1\blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX\flyquest-vs-fluxo-ancient.csv`
- round_num: `8`
- rows: `231`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 231 | 1.000 | 0.595574 | 0.605448 | -0.009874 | 108 | 123 | 0.779221 | 0.787879 |
| active/recent utility | 231 | 1.000 | 0.595574 | 0.605448 | -0.009874 | 108 | 123 | 0.779221 | 0.787879 |
| strong utility action | 165 | 0.714 | 0.614807 | 0.623675 | -0.008868 | 85 | 80 | 0.860606 | 0.872727 |
| utility damage | 26 | 0.113 | 0.678808 | 0.634378 | 0.044430 | 25 | 1 | 0.884615 | 0.846154 |
| active smoke/inferno | 165 | 0.714 | 0.614807 | 0.623675 | -0.008868 | 85 | 80 | 0.860606 | 0.872727 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 231 | 1.000 | 0.595574 | 0.605448 | -0.009874 | 108 | 123 | 0.779221 | 0.787879 |

## Active Smoke/Inferno Intervals

- `6.5s` - `41.5s`, rows `71`
- `57.0s` - `63.5s`, rows `14`
- `65.5s` - `105.0s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.0`, LSTM `0.5195`, XGBoost `0.3217`, closer `lstm`, smoke `10`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.6129`, XGBoost `0.8068`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5134`, XGBoost `0.3217`, closer `lstm`, smoke `10`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.6266`, XGBoost `0.8028`, closer `xgboost`, smoke `9`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.6324`, XGBoost `0.8048`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5016`, XGBoost `0.3448`, closer `lstm`, smoke `10`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.6569`, XGBoost `0.8125`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.6570`, XGBoost `0.8125`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.6566`, XGBoost `0.8028`, closer `xgboost`, smoke `9`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.6589`, XGBoost `0.8048`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
