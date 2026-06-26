# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-b8-inferno-RJncpU8XKWGlyue1SsisvY/virtus-pro-vs-b8-inferno.csv`
- round_num: `7`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.170866 | 0.038467 | 0.195306 | 1.000000 | 0.829134 |
| xgboost | 0.158266 | 0.036915 | 0.182896 | 1.000000 | 0.841734 |

## Closer Per Tick

- lstm: `91`
- xgboost: `139`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
