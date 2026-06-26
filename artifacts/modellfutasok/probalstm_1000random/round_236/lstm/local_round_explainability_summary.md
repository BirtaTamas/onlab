# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `23`

## Largest probability jumps

- tick `211790`, seconds `102.00`, LSTM `0.1335`, delta `-0.2909`
- tick `211406`, seconds `96.00`, LSTM `0.2065`, delta `-0.2876`
- tick `211662`, seconds `100.00`, LSTM `0.3514`, delta `-0.2783`
- tick `211630`, seconds `99.50`, LSTM `0.6297`, delta `+0.2297`
- tick `211534`, seconds `98.00`, LSTM `0.3426`, delta `+0.1956`
- tick `212238`, seconds `109.00`, LSTM `0.0168`, delta `-0.1084`
- tick `212014`, seconds `105.50`, LSTM `0.1536`, delta `+0.0744`
- tick `211598`, seconds `99.00`, LSTM `0.4000`, delta `+0.0669`
- tick `211438`, seconds `96.50`, LSTM `0.1436`, delta `-0.0629`
- tick `211822`, seconds `102.50`, LSTM `0.0868`, delta `-0.0467`

## Top 15 local ridge features

- `lag_03__CT5__flash_duration`: coefficient `-0.003144`, |coef| `0.003144`
- `lag_00__CT5__flash_duration`: coefficient `0.003068`, |coef| `0.003068`
- `lag_00__kill_diff_last_3s`: coefficient `0.002697`, |coef| `0.002697`
- `lag_00__T_kills_last_3s`: coefficient `-0.002532`, |coef| `0.002532`
- `lag_04__CT5__flash_duration`: coefficient `-0.002070`, |coef| `0.002070`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001936`, |coef| `0.001936`
- `lag_00__damage_diff_last_5s`: coefficient `0.001931`, |coef| `0.001931`
- `lag_09__CT_place_ENTRANCE`: coefficient `0.001722`, |coef| `0.001722`
- `lag_02__T_shots_fired_sum`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_04__CT_place_ENTRANCE`: coefficient `-0.001562`, |coef| `0.001562`
- `lag_04__T3__duck_amount`: coefficient `0.001527`, |coef| `0.001527`
- `lag_00__T_damage_last_5s`: coefficient `-0.001506`, |coef| `0.001506`
- `lag_06__T4__flash_duration`: coefficient `0.001485`, |coef| `0.001485`
- `lag_00__CT_flash_duration_sum`: coefficient `0.001445`, |coef| `0.001445`
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001411`, |coef| `0.001411`

## Top 10 utility ridge features

- `lag_03__CT5__flash_duration`: coefficient `-0.003144` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `0.003068` (raises CT win probability)
- `lag_04__CT5__flash_duration`: coefficient `-0.002070` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.001485` (raises CT win probability)
- `lag_00__CT_flash_duration_sum`: coefficient `0.001445` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `-0.001411` (lowers CT win probability)
- `lag_08__CT5__flash_duration`: coefficient `0.001352` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `-0.001227` (lowers CT win probability)
- `lag_12__CT5__flash_duration`: coefficient `0.001171` (raises CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `-0.001124` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002697` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002532` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001936` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001931` (raises CT win probability)
- `lag_09__CT_place_ENTRANCE`: coefficient `0.001722` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `-0.001637` (lowers CT win probability)
- `lag_04__CT_place_ENTRANCE`: coefficient `-0.001562` (lowers CT win probability)
- `lag_04__T3__duck_amount`: coefficient `0.001527` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001506` (lowers CT win probability)
- `lag_02__CT_place_ENTRANCE`: coefficient `-0.001328` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `211790`, seconds `102.00`, LSTM delta `-0.2909`

Top all feature movements:
- `lag_09__CT_place_ENTRANCE`: contribution `-0.015283`
- `lag_12__CT5__flash_duration`: contribution `-0.009510`
- `lag_15__CT5__flash_duration`: contribution `-0.008515`
- `lag_04__CT_place_LONGDOG`: contribution `-0.008421`
- `lag_00__T_kills_last_3s`: contribution `-0.008022`

Top utility-only movements:
- `lag_12__CT5__flash_duration`: contribution `-0.009510`
- `lag_15__CT5__flash_duration`: contribution `-0.008515`
- `lag_06__T4__flash_duration`: contribution `-0.007344`
- `lag_15__T1__flash_duration`: contribution `-0.003710`
- `lag_14__T4__flash_duration`: contribution `-0.003363`

### tick `211406`, seconds `96.00`, LSTM delta `-0.2876`

Top all feature movements:
- `lag_03__CT5__flash_duration`: contribution `-0.025528`
- `lag_00__CT5__flash_duration`: contribution `-0.024916`
- `lag_02__CT_place_ENTRANCE`: contribution `-0.011783`
- `lag_00__T_kills_last_3s`: contribution `-0.008022`
- `lag_00__kill_diff_last_3s`: contribution `-0.006492`

Top utility-only movements:
- `lag_03__CT5__flash_duration`: contribution `-0.025528`
- `lag_00__CT5__flash_duration`: contribution `-0.024916`
- `lag_03__T1__flash_duration`: contribution `-0.006094`
- `lag_00__CT_flash_duration_sum`: contribution `-0.005412`
- `lag_03__CT_flash_duration_sum`: contribution `-0.005284`

### tick `211662`, seconds `100.00`, LSTM delta `-0.2783`

Top all feature movements:
- `lag_08__CT5__flash_duration`: contribution `-0.010981`
- `lag_04__CT_place_ELECTRICALBOX`: contribution `-0.010756`
- `lag_11__CT5__flash_duration`: contribution `-0.009126`
- `lag_00__T_kills_last_3s`: contribution `-0.008022`
- `lag_06__T1__shots_fired`: contribution `-0.007858`

Top utility-only movements:
- `lag_08__CT5__flash_duration`: contribution `-0.010981`
- `lag_11__CT5__flash_duration`: contribution `-0.009126`
- `lag_02__T4__flash_duration`: contribution `+0.005119`

### tick `211630`, seconds `99.50`, LSTM delta `+0.2297`

Top all feature movements:
- `lag_09__CT_place_ENTRANCE`: contribution `+0.015283`
- `lag_04__CT_place_ENTRANCE`: contribution `+0.013858`
- `lag_10__CT5__flash_duration`: contribution `+0.007347`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `+0.006745`
- `lag_00__kill_diff_last_3s`: contribution `+0.006492`

Top utility-only movements:
- `lag_10__CT5__flash_duration`: contribution `+0.007347`
- `lag_01__T4__flash_duration`: contribution `+0.004600`
- `lag_10__T1__flash_duration`: contribution `+0.003444`
- `lag_09__T4__flash_duration`: contribution `+0.003247`

### tick `211534`, seconds `98.00`, LSTM delta `+0.1956`

Top all feature movements:
- `lag_04__CT5__flash_duration`: contribution `+0.016810`
- `lag_02__T_shots_fired_sum`: contribution `+0.009817`
- `lag_00__T_shots_fired_sum`: contribution `+0.008711`
- `lag_00__CT_place_ELECTRICALBOX`: contribution `+0.007436`
- `lag_06__T4__flash_duration`: contribution `+0.007344`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `+0.016810`
- `lag_06__T4__flash_duration`: contribution `+0.007344`
- `lag_07__T1__flash_duration`: contribution `+0.003827`
- `lag_04__CT_flash_duration_sum`: contribution `+0.003162`
- `lag_07__CT5__flash_duration`: contribution `+0.002757`
