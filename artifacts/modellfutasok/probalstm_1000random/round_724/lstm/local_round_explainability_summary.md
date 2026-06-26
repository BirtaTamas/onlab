# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m3-nuke.csv`
- round_num: `14`

## Largest probability jumps

- tick `116366`, seconds `81.00`, LSTM `0.6330`, delta `+0.3166`
- tick `114830`, seconds `57.00`, LSTM `0.7234`, delta `+0.2059`
- tick `115534`, seconds `68.00`, LSTM `0.4929`, delta `+0.1979`
- tick `115854`, seconds `73.00`, LSTM `0.2986`, delta `-0.1921`
- tick `115118`, seconds `61.50`, LSTM `0.5324`, delta `-0.1673`
- tick `116302`, seconds `80.00`, LSTM `0.3009`, delta `+0.1229`
- tick `115214`, seconds `63.00`, LSTM `0.3843`, delta `-0.1119`
- tick `113198`, seconds `31.50`, LSTM `0.5833`, delta `-0.0909`
- tick `115886`, seconds `73.50`, LSTM `0.2265`, delta `-0.0721`
- tick `115278`, seconds `64.00`, LSTM `0.3252`, delta `-0.0591`

## Top 15 local ridge features

- `lag_00__T_place_DECON`: coefficient `-0.005914`, |coef| `0.005914`
- `lag_02__T_place_DECON`: coefficient `-0.004750`, |coef| `0.004750`
- `lag_04__T_place_DECON`: coefficient `-0.004158`, |coef| `0.004158`
- `lag_04__T_place_HUT`: coefficient `-0.003683`, |coef| `0.003683`
- `lag_00__T_place_VENTS`: coefficient `-0.003129`, |coef| `0.003129`
- `lag_13__T_place_VENTS`: coefficient `-0.003099`, |coef| `0.003099`
- `lag_00__kill_diff_last_3s`: coefficient `0.003021`, |coef| `0.003021`
- `lag_00__damage_diff_last_5s`: coefficient `0.002896`, |coef| `0.002896`
- `lag_00__CT_kills_last_3s`: coefficient `0.002433`, |coef| `0.002433`
- `lag_13__T_place_SECRET`: coefficient `-0.002375`, |coef| `0.002375`
- `lag_10__T_place_SECRET`: coefficient `-0.002359`, |coef| `0.002359`
- `lag_01__T_place_VENTS`: coefficient `-0.002180`, |coef| `0.002180`
- `lag_15__CT_place_ADMIN`: coefficient `0.002111`, |coef| `0.002111`
- `lag_06__T_place_TUNNELS`: coefficient `-0.002083`, |coef| `0.002083`
- `lag_00__T_place_TUNNELS`: coefficient `-0.002028`, |coef| `0.002028`

## Top 10 utility ridge features

- `lag_13__CT3__flash`: coefficient `-0.001173` (lowers CT win probability)
- `lag_00__CT3__flash`: coefficient `0.001092` (raises CT win probability)
- `lag_00__T4__flash`: coefficient `-0.000967` (lowers CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000884` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000869` (lowers CT win probability)
- `lag_12__CT_flashes_last_5s`: coefficient `0.000861` (raises CT win probability)
- `lag_03__CT3__flash`: coefficient `0.000816` (raises CT win probability)
- `lag_13__T5__flash`: coefficient `-0.000763` (lowers CT win probability)
- `lag_00__T_B_site_active_smokes`: coefficient `-0.000700` (lowers CT win probability)
- `lag_13__CT_flash_inv`: coefficient `-0.000685` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_DECON`: coefficient `-0.005914` (lowers CT win probability)
- `lag_02__T_place_DECON`: coefficient `-0.004750` (lowers CT win probability)
- `lag_04__T_place_DECON`: coefficient `-0.004158` (lowers CT win probability)
- `lag_04__T_place_HUT`: coefficient `-0.003683` (lowers CT win probability)
- `lag_00__T_place_VENTS`: coefficient `-0.003129` (lowers CT win probability)
- `lag_13__T_place_VENTS`: coefficient `-0.003099` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003021` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002896` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002433` (raises CT win probability)
- `lag_13__T_place_SECRET`: coefficient `-0.002375` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `116366`, seconds `81.00`, LSTM delta `+0.3166`

Top all feature movements:
- `lag_02__T_place_DECON`: contribution `+0.076317`
- `lag_04__T_place_DECON`: contribution `+0.066793`
- `lag_13__T_place_VENTS`: contribution `+0.041792`
- `lag_06__T_place_DECON`: contribution `+0.031932`
- `lag_15__CT_place_ADMIN`: contribution `+0.014667`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `114830`, seconds `57.00`, LSTM delta `+0.2059`

Top all feature movements:
- `lag_13__T_place_SECRET`: contribution `+0.012494`
- `lag_06__CT_place_ADMIN`: contribution `+0.011046`
- `lag_01__T_place_SECRET`: contribution `+0.008776`
- `lag_00__kill_diff_last_3s`: contribution `+0.007271`
- `lag_00__CT_kills_last_3s`: contribution `+0.007024`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115534`, seconds `68.00`, LSTM delta `+0.1979`

Top all feature movements:
- `lag_04__T_place_HUT`: contribution `+0.034331`
- `lag_10__T_place_HUT`: contribution `+0.016759`
- `lag_10__T_place_SECRET`: contribution `+0.012410`
- `lag_00__kill_diff_last_3s`: contribution `+0.007271`
- `lag_00__CT_kills_last_3s`: contribution `+0.007024`

Top utility-only movements:
- `lag_13__CT3__flash`: contribution `+0.004329`

### tick `115854`, seconds `73.00`, LSTM delta `-0.1921`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `-0.095019`
- `lag_00__T_place_VENTS`: contribution `-0.042201`
- `lag_14__T_place_HUT`: contribution `-0.010419`
- `lag_00__T_place_TUNNELS`: contribution `+0.005686`
- `lag_02__CT_place_HELL`: contribution `-0.005109`

Top utility-only movements:
- `lag_10__T4__flash`: contribution `-0.001595`

### tick `115118`, seconds `61.50`, LSTM delta `-0.1673`

Top all feature movements:
- `lag_15__CT_place_ADMIN`: contribution `-0.014667`
- `lag_10__T_place_SECRET`: contribution `-0.012410`
- `lag_04__CT_place_ADMIN`: contribution `-0.008915`
- `lag_00__kill_diff_last_3s`: contribution `-0.007271`
- `lag_15__CT_place_HELL`: contribution `-0.006451`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.004030`
