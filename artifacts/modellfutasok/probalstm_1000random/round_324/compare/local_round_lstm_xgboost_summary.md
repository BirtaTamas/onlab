# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-legacy-anubis-nLMamLTYoRhlv2MuS6sSiC/betboom-vs-legacy-anubis.csv`
- round_num: `9`
- rows: `261`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.382184 | 0.153571 | 0.490326 | 0.977011 | 0.382184 |
| xgboost | 0.524086 | 0.279425 | 0.752661 | 0.272031 | 0.524086 |

## Closer Per Tick

- lstm: `261`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
