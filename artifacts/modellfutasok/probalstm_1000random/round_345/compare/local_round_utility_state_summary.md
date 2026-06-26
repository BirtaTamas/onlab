# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-3dmax-bo3-u02WLpVJ6Q22MzSL2B_-Tu/the-mongolz-vs-3dmax-m2-ancient.csv`
- round_num: `14`
- rows: `125`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 125 | 1.000 | 0.360422 | 0.395246 | -0.034825 | 91 | 34 | 0.712000 | 0.704000 |
| active/recent utility | 125 | 1.000 | 0.360422 | 0.395246 | -0.034825 | 91 | 34 | 0.712000 | 0.704000 |
| strong utility action | 112 | 0.896 | 0.331616 | 0.369094 | -0.037478 | 84 | 28 | 0.794643 | 0.785714 |
| utility damage | 10 | 0.080 | 0.108530 | 0.168821 | -0.060291 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 112 | 0.896 | 0.331616 | 0.369094 | -0.037478 | 84 | 28 | 0.794643 | 0.785714 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 125 | 1.000 | 0.360422 | 0.395246 | -0.034825 | 91 | 34 | 0.712000 | 0.704000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `62.0s`, rows `112`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `62.0`, LSTM `0.7182`, XGBoost `0.9060`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.7376`, XGBoost `0.9090`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.7433`, XGBoost `0.9053`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.7486`, XGBoost `0.9066`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.7558`, XGBoost `0.9062`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.7569`, XGBoost `0.9066`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.7641`, XGBoost `0.9066`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.2365`, XGBoost `0.3628`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.4265`, XGBoost `0.5480`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1779`, XGBoost `0.2954`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
