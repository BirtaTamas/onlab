# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `6`
- rows: `193`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 193 | 1.000 | 0.076932 | 0.162393 | -0.085461 | 193 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 193 | 1.000 | 0.076932 | 0.162393 | -0.085461 | 193 | 0 | 1.000000 | 1.000000 |
| strong utility action | 174 | 0.902 | 0.078524 | 0.160267 | -0.081743 | 174 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.052 | 0.184340 | 0.214873 | -0.030534 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 164 | 0.850 | 0.073892 | 0.157515 | -0.083623 | 164 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.052 | 0.154482 | 0.205400 | -0.050918 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 193 | 1.000 | 0.076932 | 0.162393 | -0.085461 | 193 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `48.5s`, rows `82`
- `55.5s` - `96.0s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.1985`, XGBoost `0.3644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0303`, XGBoost `0.1736`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0348`, XGBoost `0.1736`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1312`, XGBoost `0.2630`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0409`, XGBoost `0.1722`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0430`, XGBoost `0.1736`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0426`, XGBoost `0.1731`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0450`, XGBoost `0.1722`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1359`, XGBoost `0.2630`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0465`, XGBoost `0.1722`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
