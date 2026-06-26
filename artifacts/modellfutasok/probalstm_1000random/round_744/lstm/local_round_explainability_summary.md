# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `131069`, seconds `33.50`, LSTM `0.6472`, delta `+0.3330`
- tick `131229`, seconds `36.00`, LSTM `0.3466`, delta `-0.2124`
- tick `131165`, seconds `35.00`, LSTM `0.5744`, delta `-0.2048`
- tick `131005`, seconds `32.50`, LSTM `0.2685`, delta `+0.1996`
- tick `131485`, seconds `40.00`, LSTM `0.1086`, delta `-0.1879`
- tick `133309`, seconds `68.50`, LSTM `0.1212`, delta `+0.0961`
- tick `131101`, seconds `34.00`, LSTM `0.7279`, delta `+0.0807`
- tick `131869`, seconds `46.00`, LSTM `0.0225`, delta `-0.0624`
- tick `131325`, seconds `37.50`, LSTM `0.2959`, delta `-0.0569`
- tick `128957`, seconds `0.50`, LSTM `0.0406`, delta `-0.0526`

## Top 15 local ridge features

- `lag_15__CT_place_MAINHALL`: coefficient `-0.005205`, |coef| `0.005205`
- `lag_00__kill_diff_last_3s`: coefficient `0.003998`, |coef| `0.003998`
- `lag_00__damage_diff_last_5s`: coefficient `0.002984`, |coef| `0.002984`
- `lag_00__CT_kills_last_3s`: coefficient `0.002866`, |coef| `0.002866`
- `lag_13__CT_place_MAINHALL`: coefficient `-0.002691`, |coef| `0.002691`
- `lag_14__T_place_RAMP`: coefficient `0.002468`, |coef| `0.002468`
- `lag_13__T_place_RAMP`: coefficient `0.002404`, |coef| `0.002404`
- `lag_06__T1__is_scoped`: coefficient `0.002254`, |coef| `0.002254`
- `lag_00__T_kills_last_3s`: coefficient `-0.002117`, |coef| `0.002117`
- `lag_12__T_place_RAMP`: coefficient `0.002072`, |coef| `0.002072`
- `lag_14__CT_place_MAINHALL`: coefficient `-0.002058`, |coef| `0.002058`
- `lag_01__T_place_RAMP`: coefficient `-0.001855`, |coef| `0.001855`
- `lag_03__T1__is_scoped`: coefficient `0.001775`, |coef| `0.001775`
- `lag_00__CT_damage_last_5s`: coefficient `0.001697`, |coef| `0.001697`
- `lag_07__CT2__is_walking`: coefficient `0.001665`, |coef| `0.001665`

## Top 10 utility ridge features

- `lag_03__T5__molly`: coefficient `-0.001053` (lowers CT win probability)
- `lag_02__T3__smoke`: coefficient `-0.001031` (lowers CT win probability)
- `lag_12__T1__smoke`: coefficient `-0.001006` (lowers CT win probability)
- `lag_02__T3__utility_total`: coefficient `-0.000780` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000706` (lowers CT win probability)
- `lag_02__T3__flash`: coefficient `-0.000701` (lowers CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `0.000701` (raises CT win probability)
- `lag_01__T5__molly`: coefficient `-0.000697` (lowers CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000661` (lowers CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `-0.000643` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_MAINHALL`: coefficient `-0.005205` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003998` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002984` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002866` (raises CT win probability)
- `lag_13__CT_place_MAINHALL`: coefficient `-0.002691` (lowers CT win probability)
- `lag_14__T_place_RAMP`: coefficient `0.002468` (raises CT win probability)
- `lag_13__T_place_RAMP`: coefficient `0.002404` (raises CT win probability)
- `lag_06__T1__is_scoped`: coefficient `0.002254` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002117` (lowers CT win probability)
- `lag_12__T_place_RAMP`: coefficient `0.002072` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `131069`, seconds `33.50`, LSTM delta `+0.3330`

Top all feature movements:
- `lag_15__CT_place_MAINHALL`: contribution `+0.086155`
- `lag_06__T1__is_scoped`: contribution `+0.012880`
- `lag_00__kill_diff_last_3s`: contribution `+0.009622`
- `lag_00__CT_kills_last_3s`: contribution `+0.008273`
- `lag_12__T_place_RAMP`: contribution `+0.007327`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131229`, seconds `36.00`, LSTM delta `-0.2124`

Top all feature movements:
- `lag_03__T1__is_scoped`: contribution `-0.010142`
- `lag_00__kill_diff_last_3s`: contribution `-0.009622`
- `lag_12__T_place_RAMP`: contribution `-0.007327`
- `lag_00__damage_diff_last_5s`: contribution `-0.006733`
- `lag_00__T_kills_last_3s`: contribution `-0.006708`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131165`, seconds `35.00`, LSTM delta `-0.2048`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009622`
- `lag_01__T1__is_scoped`: contribution `-0.008956`
- `lag_13__T_place_RAMP`: contribution `-0.008501`
- `lag_15__T_shots_fired_sum`: contribution `-0.007086`
- `lag_01__CT_place_SIDEHALL`: contribution `-0.006943`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131005`, seconds `32.50`, LSTM delta `+0.1996`

Top all feature movements:
- `lag_13__CT_place_MAINHALL`: contribution `+0.044543`
- `lag_00__kill_diff_last_3s`: contribution `+0.009622`
- `lag_14__T_place_RAMP`: contribution `+0.008729`
- `lag_00__CT_kills_last_3s`: contribution `+0.008273`
- `lag_00__damage_diff_last_5s`: contribution `+0.006733`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `131485`, seconds `40.00`, LSTM delta `-0.1879`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009622`
- `lag_14__T_place_RAMP`: contribution `-0.008729`
- `lag_08__CT_place_HOUSE`: contribution `-0.007443`
- `lag_00__T_kills_last_3s`: contribution `-0.006708`
- `lag_11__CT_place_SIDEHALL`: contribution `-0.006386`

Top utility-only movements:
- No utility movement among the top local contributors.
