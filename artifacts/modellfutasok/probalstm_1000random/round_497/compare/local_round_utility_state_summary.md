# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `17`
- rows: `160`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 160 | 1.000 | 0.572038 | 0.555239 | 0.016799 | 115 | 45 | 0.656250 | 0.625000 |
| active/recent utility | 160 | 1.000 | 0.572038 | 0.555239 | 0.016799 | 115 | 45 | 0.656250 | 0.625000 |
| strong utility action | 119 | 0.744 | 0.608393 | 0.566442 | 0.041951 | 92 | 27 | 0.705882 | 0.647059 |
| utility damage | 10 | 0.062 | 0.768899 | 0.669360 | 0.099539 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 119 | 0.744 | 0.608393 | 0.566442 | 0.041951 | 92 | 27 | 0.705882 | 0.647059 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 160 | 1.000 | 0.572038 | 0.555239 | 0.016799 | 115 | 45 | 0.656250 | 0.625000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `61.5s`, rows `105`
- `64.0s` - `70.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.5`, LSTM `0.6498`, XGBoost `0.5034`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6975`, XGBoost `0.5566`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.4570`, XGBoost `0.3185`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.4804`, XGBoost `0.3448`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6839`, XGBoost `0.5534`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6227`, XGBoost `0.5023`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.4616`, XGBoost `0.3437`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.4537`, XGBoost `0.3363`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.4436`, XGBoost `0.5579`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7331`, XGBoost `0.6204`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
