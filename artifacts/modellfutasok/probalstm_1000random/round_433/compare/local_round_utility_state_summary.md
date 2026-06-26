# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `1`
- rows: `130`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 130 | 1.000 | 0.108186 | 0.113349 | -0.005163 | 100 | 30 | 0.838462 | 1.000000 |
| active/recent utility | 130 | 1.000 | 0.108186 | 0.113349 | -0.005163 | 100 | 30 | 0.838462 | 1.000000 |
| strong utility action | 87 | 0.669 | 0.016047 | 0.028692 | -0.012645 | 87 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 87 | 0.669 | 0.016047 | 0.028692 | -0.012645 | 87 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 130 | 1.000 | 0.108186 | 0.113349 | -0.005163 | 100 | 30 | 0.838462 | 1.000000 |

## Active Smoke/Inferno Intervals

- `12.0s` - `55.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.5`, LSTM `0.0390`, XGBoost `0.1204`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.3404`, XGBoost `0.4177`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0481`, XGBoost `0.1253`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.3971`, XGBoost `0.4626`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0655`, XGBoost `0.1175`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0735`, XGBoost `0.1253`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0695`, XGBoost `0.1209`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0714`, XGBoost `0.1147`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0238`, XGBoost `0.0624`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0241`, XGBoost `0.0579`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
