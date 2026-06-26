# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `19`

## Largest probability jumps

- tick `158304`, seconds `56.50`, LSTM `0.7771`, delta `+0.3408`
- tick `158240`, seconds `55.50`, LSTM `0.3852`, delta `-0.1732`
- tick `158176`, seconds `54.50`, LSTM `0.5142`, delta `+0.1646`
- tick `158336`, seconds `57.00`, LSTM `0.8990`, delta `+0.1219`
- tick `158816`, seconds `64.50`, LSTM `0.9600`, delta `+0.0597`
- tick `158272`, seconds `56.00`, LSTM `0.4362`, delta `+0.0510`
- tick `155968`, seconds `20.00`, LSTM `0.4589`, delta `+0.0447`
- tick `158208`, seconds `55.00`, LSTM `0.5584`, delta `+0.0441`
- tick `155168`, seconds `7.50`, LSTM `0.4163`, delta `-0.0301`
- tick `156032`, seconds `21.00`, LSTM `0.4418`, delta `-0.0269`

## Top 15 local ridge features

- `lag_00__T_place_JUNGLE`: coefficient `0.002828`, |coef| `0.002828`
- `lag_02__CT_place_STAIRS`: coefficient `-0.002204`, |coef| `0.002204`
- `lag_11__T_place_CONNECTOR`: coefficient `0.002203`, |coef| `0.002203`
- `lag_00__kill_diff_last_3s`: coefficient `0.002101`, |coef| `0.002101`
- `lag_00__CT_kills_last_3s`: coefficient `0.002092`, |coef| `0.002092`
- `lag_00__CT4__flash_duration`: coefficient `0.001952`, |coef| `0.001952`
- `lag_06__T_place_CONNECTOR`: coefficient `0.001912`, |coef| `0.001912`
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001910`, |coef| `0.001910`
- `lag_10__T_place_CONNECTOR`: coefficient `0.001814`, |coef| `0.001814`
- `lag_15__T4__duck_amount`: coefficient `0.001724`, |coef| `0.001724`
- `lag_00__T1__flash_duration`: coefficient `0.001534`, |coef| `0.001534`
- `lag_07__T_place_CONNECTOR`: coefficient `0.001526`, |coef| `0.001526`
- `lag_00__CT_damage_last_5s`: coefficient `0.001508`, |coef| `0.001508`
- `lag_12__CT4__is_walking`: coefficient `-0.001467`, |coef| `0.001467`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001417`, |coef| `0.001417`

## Top 10 utility ridge features

- `lag_00__CT4__flash_duration`: coefficient `0.001952` (raises CT win probability)
- `lag_00__T1__flash_duration`: coefficient `0.001534` (raises CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.001127` (raises CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000829` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000752` (lowers CT win probability)
- `lag_04__T5__smoke`: coefficient `-0.000692` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.000669` (lowers CT win probability)
- `lag_00__T3__flash`: coefficient `-0.000653` (lowers CT win probability)
- `lag_01__CT5__flash`: coefficient `-0.000637` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000606` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_JUNGLE`: coefficient `0.002828` (raises CT win probability)
- `lag_02__CT_place_STAIRS`: coefficient `-0.002204` (lowers CT win probability)
- `lag_11__T_place_CONNECTOR`: coefficient `0.002203` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002101` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002092` (raises CT win probability)
- `lag_06__T_place_CONNECTOR`: coefficient `0.001912` (raises CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `-0.001910` (lowers CT win probability)
- `lag_10__T_place_CONNECTOR`: coefficient `0.001814` (raises CT win probability)
- `lag_15__T4__duck_amount`: coefficient `0.001724` (raises CT win probability)
- `lag_07__T_place_CONNECTOR`: coefficient `0.001526` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `158304`, seconds `56.50`, LSTM delta `+0.3408`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `+0.036631`
- `lag_02__CT_place_STAIRS`: contribution `+0.017152`
- `lag_11__T_place_CONNECTOR`: contribution `+0.010669`
- `lag_00__CT4__flash_duration`: contribution `+0.010018`
- `lag_10__T_place_CONNECTOR`: contribution `+0.008782`

Top utility-only movements:
- `lag_00__CT4__flash_duration`: contribution `+0.010018`
- `lag_00__T1__flash_duration`: contribution `+0.007807`

### tick `158240`, seconds `55.50`, LSTM delta `-0.1732`

Top all feature movements:
- `lag_00__CT_place_STAIRS`: contribution `-0.007968`
- `lag_15__T4__duck_amount`: contribution `-0.006374`
- `lag_00__kill_diff_last_3s`: contribution `-0.005058`
- `lag_04__T4__is_scoped`: contribution `-0.003998`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003938`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `158176`, seconds `54.50`, LSTM delta `+0.1646`

Top all feature movements:
- `lag_06__T_place_CONNECTOR`: contribution `+0.009259`
- `lag_00__T_place_CONNECTOR`: contribution `+0.009248`
- `lag_07__T_place_CONNECTOR`: contribution `+0.007389`
- `lag_00__CT_kills_last_3s`: contribution `+0.006041`
- `lag_00__kill_diff_last_3s`: contribution `+0.005058`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `158336`, seconds `57.00`, LSTM delta `+0.1219`

Top all feature movements:
- `lag_01__T_place_JUNGLE`: contribution `+0.015938`
- `lag_11__T_place_CONNECTOR`: contribution `+0.010669`
- `lag_03__CT_place_STAIRS`: contribution `+0.007572`
- `lag_00__CT_kills_last_3s`: contribution `+0.006041`
- `lag_00__CT_shots_fired_sum`: contribution `-0.005907`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.005784`
- `lag_01__T1__flash_duration`: contribution `+0.002733`

### tick `158816`, seconds `64.50`, LSTM delta `+0.0597`

Top all feature movements:
- `lag_00__T_place_JUNGLE`: contribution `-0.036631`
- `lag_09__T_place_JUNGLE`: contribution `+0.014040`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007876`
- `lag_00__CT_kills_last_3s`: contribution `+0.006041`
- `lag_00__kill_diff_last_3s`: contribution `+0.005058`

Top utility-only movements:
- `lag_07__T1__flash_duration`: contribution `+0.001727`
- `lag_00__T1__utility_total`: contribution `+0.001328`
- `lag_00__T1__flash`: contribution `+0.001046`
