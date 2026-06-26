# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `11`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.713207 | 0.659806 | 0.053401 | 128 | 45 | 1.000000 | 1.000000 |
| active/recent utility | 173 | 1.000 | 0.713207 | 0.659806 | 0.053401 | 128 | 45 | 1.000000 | 1.000000 |
| strong utility action | 157 | 0.908 | 0.715127 | 0.659823 | 0.055304 | 116 | 41 | 1.000000 | 1.000000 |
| utility damage | 33 | 0.191 | 0.699798 | 0.631517 | 0.068281 | 33 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 157 | 0.908 | 0.715127 | 0.659823 | 0.055304 | 116 | 41 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 173 | 1.000 | 0.713207 | 0.659806 | 0.053401 | 128 | 45 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `84.0s`, rows `157`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.8033`, XGBoost `0.5745`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.7939`, XGBoost `0.5743`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.7935`, XGBoost `0.5745`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.7899`, XGBoost `0.5743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.7823`, XGBoost `0.5745`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.7802`, XGBoost `0.5743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.7398`, XGBoost `0.5364`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.7772`, XGBoost `0.5743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.7775`, XGBoost `0.5746`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.7769`, XGBoost `0.5743`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
