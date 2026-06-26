# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `3`
- rows: `109`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 109 | 1.000 | 0.400395 | 0.371964 | 0.028431 | 52 | 57 | 0.422018 | 0.550459 |
| active/recent utility | 109 | 1.000 | 0.400395 | 0.371964 | 0.028431 | 52 | 57 | 0.422018 | 0.550459 |
| strong utility action | 92 | 0.844 | 0.372772 | 0.351734 | 0.021038 | 52 | 40 | 0.500000 | 0.510870 |
| utility damage | 21 | 0.193 | 0.542364 | 0.458134 | 0.084230 | 4 | 17 | 0.333333 | 0.380952 |
| active smoke/inferno | 92 | 0.844 | 0.372772 | 0.351734 | 0.021038 | 52 | 40 | 0.500000 | 0.510870 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 109 | 1.000 | 0.400395 | 0.371964 | 0.028431 | 52 | 57 | 0.422018 | 0.550459 |

## Active Smoke/Inferno Intervals

- `8.5s` - `54.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.0`, LSTM `0.4945`, XGBoost `0.3223`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.4619`, XGBoost `0.3111`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.4708`, XGBoost `0.3219`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6219`, XGBoost `0.4758`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `22.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.4460`, XGBoost `0.3111`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.4129`, XGBoost `0.2855`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `18.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.4050`, XGBoost `0.2897`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.4347`, XGBoost `0.3221`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6449`, XGBoost `0.5346`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `22.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.4041`, XGBoost `0.2958`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
