# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `7`
- rows: `177`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.056584 | 0.010562 | 0.062783 | 1.000000 | 0.056584 |
| xgboost | 0.130641 | 0.037122 | 0.154407 | 1.000000 | 0.130641 |

## Closer Per Tick

- lstm: `177`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
