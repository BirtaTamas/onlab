# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `10`

## Largest probability jumps

- tick `61532`, seconds `65.50`, LSTM `0.1305`, delta `-0.3006`
- tick `60988`, seconds `57.00`, LSTM `0.5314`, delta `-0.1465`
- tick `61916`, seconds `71.50`, LSTM `0.0350`, delta `-0.1427`
- tick `61148`, seconds `59.50`, LSTM `0.5242`, delta `-0.0904`
- tick `61116`, seconds `59.00`, LSTM `0.6146`, delta `+0.0852`
- tick `61500`, seconds `65.00`, LSTM `0.4311`, delta `+0.0821`
- tick `60732`, seconds `53.00`, LSTM `0.6839`, delta `-0.0817`
- tick `61692`, seconds `68.00`, LSTM `0.1363`, delta `-0.0800`
- tick `61436`, seconds `64.00`, LSTM `0.3667`, delta `-0.0729`
- tick `62396`, seconds `79.00`, LSTM `0.0798`, delta `+0.0689`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002487`, |coef| `0.002487`
- `lag_00__T_place_DECON`: coefficient `-0.001983`, |coef| `0.001983`
- `lag_12__CT_place_VENTS`: coefficient `0.001938`, |coef| `0.001938`
- `lag_00__T_place_HEAVEN`: coefficient `-0.001927`, |coef| `0.001927`
- `lag_12__T4__shots_fired`: coefficient `-0.001714`, |coef| `0.001714`
- `lag_03__T_place_DECON`: coefficient `-0.001712`, |coef| `0.001712`
- `lag_13__T_place_OBSERVATION`: coefficient `0.001672`, |coef| `0.001672`
- `lag_06__T4__shots_fired`: coefficient `-0.001625`, |coef| `0.001625`
- `lag_14__T_place_OBSERVATION`: coefficient `0.001542`, |coef| `0.001542`
- `lag_02__T3__duck_amount`: coefficient `0.001440`, |coef| `0.001440`
- `lag_07__T_place_DECON`: coefficient `-0.001438`, |coef| `0.001438`
- `lag_00__T4__shots_fired`: coefficient `-0.001429`, |coef| `0.001429`
- `lag_11__T_place_SQUEAKY`: coefficient `0.001388`, |coef| `0.001388`
- `lag_12__CT_place_ADMIN`: coefficient `-0.001380`, |coef| `0.001380`
- `lag_04__T_shots_fired_sum`: coefficient `0.001340`, |coef| `0.001340`

## Top 10 utility ridge features

- `lag_00__CT3__molly`: coefficient `0.000879` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000785` (raises CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000781` (raises CT win probability)
- `lag_12__CT1__smoke`: coefficient `0.000572` (raises CT win probability)
- `lag_02__T3__smoke`: coefficient `0.000571` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000570` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000555` (raises CT win probability)
- `lag_00__CT_utility_inv`: coefficient `0.000541` (raises CT win probability)
- `lag_01__CT4__molly`: coefficient `0.000526` (raises CT win probability)
- `lag_12__CT1__utility_total`: coefficient `0.000497` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002487` (raises CT win probability)
- `lag_00__T_place_DECON`: coefficient `-0.001983` (lowers CT win probability)
- `lag_12__CT_place_VENTS`: coefficient `0.001938` (raises CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.001927` (lowers CT win probability)
- `lag_12__T4__shots_fired`: coefficient `-0.001714` (lowers CT win probability)
- `lag_03__T_place_DECON`: coefficient `-0.001712` (lowers CT win probability)
- `lag_13__T_place_OBSERVATION`: coefficient `0.001672` (raises CT win probability)
- `lag_06__T4__shots_fired`: coefficient `-0.001625` (lowers CT win probability)
- `lag_14__T_place_OBSERVATION`: coefficient `0.001542` (raises CT win probability)
- `lag_02__T3__duck_amount`: coefficient `0.001440` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `61532`, seconds `65.50`, LSTM delta `-0.3006`

Top all feature movements:
- `lag_03__T_place_DECON`: contribution `-0.027503`
- `lag_07__T_place_DECON`: contribution `-0.023103`
- `lag_12__CT_place_VENTS`: contribution `-0.016257`
- `lag_02__T_place_DECON`: contribution `-0.013580`
- `lag_12__CT_place_ADMIN`: contribution `-0.009588`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `60988`, seconds `57.00`, LSTM delta `-0.1465`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.012096`
- `lag_00__T_shots_fired_sum`: contribution `-0.009394`
- `lag_00__CT_place_LOCKERROOM`: contribution `-0.009337`
- `lag_03__CT_place_LOCKERROOM`: contribution `-0.007609`
- `lag_00__CT_place_VENTS`: contribution `-0.007286`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61916`, seconds `71.50`, LSTM delta `-0.1427`

Top all feature movements:
- `lag_07__T_place_DECON`: contribution `-0.023103`
- `lag_13__T_place_DECON`: contribution `-0.017992`
- `lag_15__T_place_DECON`: contribution `-0.012098`
- `lag_12__T4__shots_fired`: contribution `-0.005293`
- `lag_14__T_place_DECON`: contribution `+0.004447`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61148`, seconds `59.50`, LSTM delta `-0.0904`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `-0.017082`
- `lag_00__CT_place_VENTS`: contribution `+0.007286`
- `lag_06__T4__shots_fired`: contribution `-0.007027`
- `lag_05__CT_place_LOCKERROOM`: contribution `-0.006934`
- `lag_05__CT_place_HELL`: contribution `-0.005284`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61116`, seconds `59.00`, LSTM delta `+0.0852`

Top all feature movements:
- `lag_12__CT_place_VENTS`: contribution `+0.016257`
- `lag_04__T_shots_fired_sum`: contribution `+0.010048`
- `lag_04__CT_place_LOCKERROOM`: contribution `+0.008539`
- `lag_03__T4__shots_fired`: contribution `+0.008403`
- `lag_03__T_shots_fired_sum`: contribution `+0.007900`

Top utility-only movements:
- No utility movement among the top local contributors.
