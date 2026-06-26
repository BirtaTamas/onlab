# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `2`

## Largest probability jumps

- tick `17621`, seconds `60.00`, LSTM `0.3478`, delta `-0.1964`
- tick `17909`, seconds `64.50`, LSTM `0.1340`, delta `-0.1815`
- tick `14165`, seconds `6.00`, LSTM `0.2601`, delta `+0.1402`
- tick `17653`, seconds `60.50`, LSTM `0.2282`, delta `-0.1197`
- tick `17877`, seconds `64.00`, LSTM `0.3155`, delta `+0.1190`
- tick `15253`, seconds `23.00`, LSTM `0.6242`, delta `+0.1142`
- tick `13813`, seconds `0.50`, LSTM `0.1101`, delta `-0.0880`
- tick `17941`, seconds `65.00`, LSTM `0.0487`, delta `-0.0853`
- tick `15797`, seconds `31.50`, LSTM `0.6828`, delta `+0.0600`
- tick `17685`, seconds `61.00`, LSTM `0.1708`, delta `-0.0574`

## Top 15 local ridge features

- `lag_01__CT_place_SIDE`: coefficient `0.002593`, |coef| `0.002593`
- `lag_10__CT_place_SIDE`: coefficient `0.002529`, |coef| `0.002529`
- `lag_03__CT_place_SIDE`: coefficient `-0.002261`, |coef| `0.002261`
- `lag_07__CT_place_LONGDOORS`: coefficient `0.002017`, |coef| `0.002017`
- `lag_00__kill_diff_last_3s`: coefficient `0.001924`, |coef| `0.001924`
- `lag_04__CT_place_SIDE`: coefficient `-0.001889`, |coef| `0.001889`
- `lag_00__T_place_ARAMP`: coefficient `-0.001783`, |coef| `0.001783`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.001692`, |coef| `0.001692`
- `lag_01__T_place_TSPAWN`: coefficient `-0.001691`, |coef| `0.001691`
- `lag_05__CT_place_LONGDOORS`: coefficient `0.001691`, |coef| `0.001691`
- `lag_09__CT_place_LONGDOORS`: coefficient `0.001682`, |coef| `0.001682`
- `lag_05__CT_place_MIDDOORS`: coefficient `0.001561`, |coef| `0.001561`
- `lag_10__CT_place_LONGDOORS`: coefficient `0.001469`, |coef| `0.001469`
- `lag_06__CT_place_LONGDOORS`: coefficient `0.001463`, |coef| `0.001463`
- `lag_00__CT_place_SIDE`: coefficient `0.001420`, |coef| `0.001420`

## Top 10 utility ridge features

- `lag_12__T_flashes_last_5s`: coefficient `-0.001016` (lowers CT win probability)
- `lag_01__T2__utility_total`: coefficient `-0.000803` (lowers CT win probability)
- `lag_01__T1__molly`: coefficient `-0.000781` (lowers CT win probability)
- `lag_01__T2__molly`: coefficient `-0.000780` (lowers CT win probability)
- `lag_01__T2__smoke`: coefficient `-0.000759` (lowers CT win probability)
- `lag_12__T_smoke_inv`: coefficient `0.000753` (raises CT win probability)
- `lag_12__CT5__smoke`: coefficient `0.000684` (raises CT win probability)
- `lag_12__T3__smoke`: coefficient `0.000672` (raises CT win probability)
- `lag_15__T3__smoke`: coefficient `0.000669` (raises CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.000661` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_SIDE`: coefficient `0.002593` (raises CT win probability)
- `lag_10__CT_place_SIDE`: coefficient `0.002529` (raises CT win probability)
- `lag_03__CT_place_SIDE`: coefficient `-0.002261` (lowers CT win probability)
- `lag_07__CT_place_LONGDOORS`: coefficient `0.002017` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001924` (raises CT win probability)
- `lag_04__CT_place_SIDE`: coefficient `-0.001889` (lowers CT win probability)
- `lag_00__T_place_ARAMP`: coefficient `-0.001783` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.001692` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.001691` (lowers CT win probability)
- `lag_05__CT_place_LONGDOORS`: coefficient `0.001691` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `17621`, seconds `60.00`, LSTM delta `-0.1964`

Top all feature movements:
- `lag_01__CT_place_SIDE`: contribution `-0.084474`
- `lag_02__CT_place_SIDE`: contribution `-0.036952`
- `lag_05__CT_place_LONGDOORS`: contribution `-0.007407`
- `lag_10__CT_place_LONGDOORS`: contribution `-0.006435`
- `lag_06__CT_place_LONGDOORS`: contribution `-0.006406`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17909`, seconds `64.50`, LSTM delta `-0.1815`

Top all feature movements:
- `lag_10__CT_place_SIDE`: contribution `-0.082389`
- `lag_11__CT_place_SIDE`: contribution `-0.017668`
- `lag_08__T_place_ARAMP`: contribution `-0.005428`
- `lag_00__kill_diff_last_3s`: contribution `-0.004630`
- `lag_01__T_place_ARAMP`: contribution `+0.003866`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `14165`, seconds `6.00`, LSTM delta `+0.1402`

Top all feature movements:
- `lag_05__CT_place_MIDDOORS`: contribution `+0.009012`
- `lag_00__kill_diff_last_3s`: contribution `+0.004630`
- `lag_02__T_place_OUTSIDETUNNEL`: contribution `+0.004163`
- `lag_00__CT_kills_last_3s`: contribution `+0.003855`
- `lag_05__CT_macro_MID`: contribution `+0.003677`

Top utility-only movements:
- `lag_12__T_smoke_inv`: contribution `+0.001717`

### tick `17653`, seconds `60.50`, LSTM delta `-0.1197`

Top all feature movements:
- `lag_03__CT_place_SIDE`: contribution `-0.073666`
- `lag_02__CT_place_SIDE`: contribution `+0.036952`
- `lag_00__T_place_ARAMP`: contribution `-0.016131`
- `lag_07__CT_place_LONGDOORS`: contribution `-0.008833`
- `lag_06__CT_place_LONGDOORS`: contribution `-0.006406`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `17877`, seconds `64.00`, LSTM delta `+0.1190`

Top all feature movements:
- `lag_10__CT_place_SIDE`: contribution `+0.082389`
- `lag_00__T_place_ARAMP`: contribution `+0.016131`
- `lag_00__kill_diff_last_3s`: contribution `+0.004630`
- `lag_00__CT_kills_last_3s`: contribution `+0.003855`
- `lag_00__damage_diff_last_5s`: contribution `-0.002823`

Top utility-only movements:
- No utility movement among the top local contributors.
