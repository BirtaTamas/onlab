# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `8`

## Largest probability jumps

- tick `64027`, seconds `15.00`, LSTM `0.2944`, delta `-0.2905`
- tick `64091`, seconds `16.00`, LSTM `0.0799`, delta `-0.1706`
- tick `64123`, seconds `16.50`, LSTM `0.0195`, delta `-0.0605`
- tick `64059`, seconds `15.50`, LSTM `0.2505`, delta `-0.0439`
- tick `63323`, seconds `4.00`, LSTM `0.5784`, delta `-0.0296`
- tick `63867`, seconds `12.50`, LSTM `0.5909`, delta `-0.0227`
- tick `63611`, seconds `8.50`, LSTM `0.5967`, delta `+0.0223`
- tick `63675`, seconds `9.50`, LSTM `0.6164`, delta `+0.0151`
- tick `63707`, seconds `10.00`, LSTM `0.6029`, delta `-0.0135`
- tick `63547`, seconds `7.50`, LSTM `0.5692`, delta `+0.0107`

## Top 15 local ridge features

- `lag_11__CT_place_ELECTRICALBOX`: coefficient `-0.001601`, |coef| `0.001601`
- `lag_05__CT_place_ELECTRICALBOX`: coefficient `0.001585`, |coef| `0.001585`
- `lag_08__CT_place_ELECTRICALBOX`: coefficient `0.001515`, |coef| `0.001515`
- `lag_13__CT_place_ELECTRICALBOX`: coefficient `-0.001291`, |coef| `0.001291`
- `lag_01__T_place_DUMPSTER`: coefficient `-0.001247`, |coef| `0.001247`
- `lag_02__CT1__flash_duration`: coefficient `-0.001222`, |coef| `0.001222`
- `lag_00__CT1__flash_duration`: coefficient `-0.001143`, |coef| `0.001143`
- `lag_12__CT_place_ELECTRICALBOX`: coefficient `-0.001133`, |coef| `0.001133`
- `lag_10__T3__flash_duration`: coefficient `-0.001128`, |coef| `0.001128`
- `lag_10__T_place_DUMPSTER`: coefficient `0.001053`, |coef| `0.001053`
- `lag_04__CT2__flash_duration`: coefficient `-0.001043`, |coef| `0.001043`
- `lag_13__T_place_DUMPSTER`: coefficient `-0.000994`, |coef| `0.000994`
- `lag_15__T_place_DUMPSTER`: coefficient `-0.000894`, |coef| `0.000894`
- `lag_00__CT5__flash_duration`: coefficient `-0.000874`, |coef| `0.000874`
- `lag_00__T_kills_last_3s`: coefficient `-0.000859`, |coef| `0.000859`

## Top 10 utility ridge features

- `lag_02__CT1__flash_duration`: coefficient `-0.001222` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.001143` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `-0.001128` (lowers CT win probability)
- `lag_04__CT2__flash_duration`: coefficient `-0.001043` (lowers CT win probability)
- `lag_00__CT5__flash_duration`: coefficient `-0.000874` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `-0.000857` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.000744` (lowers CT win probability)
- `lag_12__T3__flash_duration`: coefficient `-0.000685` (lowers CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000675` (raises CT win probability)
- `lag_10__CT1__flash_duration`: coefficient `-0.000652` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_ELECTRICALBOX`: coefficient `-0.001601` (lowers CT win probability)
- `lag_05__CT_place_ELECTRICALBOX`: coefficient `0.001585` (raises CT win probability)
- `lag_08__CT_place_ELECTRICALBOX`: coefficient `0.001515` (raises CT win probability)
- `lag_13__CT_place_ELECTRICALBOX`: coefficient `-0.001291` (lowers CT win probability)
- `lag_01__T_place_DUMPSTER`: coefficient `-0.001247` (lowers CT win probability)
- `lag_12__CT_place_ELECTRICALBOX`: coefficient `-0.001133` (lowers CT win probability)
- `lag_10__T_place_DUMPSTER`: coefficient `0.001053` (raises CT win probability)
- `lag_13__T_place_DUMPSTER`: coefficient `-0.000994` (lowers CT win probability)
- `lag_15__T_place_DUMPSTER`: coefficient `-0.000894` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000859` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `64027`, seconds `15.00`, LSTM delta `-0.2905`

Top all feature movements:
- `lag_11__CT_place_ELECTRICALBOX`: contribution `-0.018613`
- `lag_05__CT_place_ELECTRICALBOX`: contribution `-0.018430`
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.017611`
- `lag_01__T_place_DUMPSTER`: contribution `-0.011335`
- `lag_10__T_place_DUMPSTER`: contribution `-0.009577`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `-0.007972`
- `lag_04__CT2__flash_duration`: contribution `-0.007327`
- `lag_10__T3__flash_duration`: contribution `-0.007276`
- `lag_00__CT5__flash_duration`: contribution `-0.004429`
- `lag_00__CT_flash_duration_sum`: contribution `-0.002773`

### tick `64091`, seconds `16.00`, LSTM delta `-0.1706`

Top all feature movements:
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.015011`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `-0.013170`
- `lag_02__CT1__flash_duration`: contribution `-0.008528`
- `lag_15__T_place_DUMPSTER`: contribution `-0.008131`
- `lag_10__CT_place_ELECTRICALBOX`: contribution `+0.007985`

Top utility-only movements:
- `lag_02__CT1__flash_duration`: contribution `-0.008528`
- `lag_00__CT1__flash_duration`: contribution `+0.007972`
- `lag_06__CT2__flash_duration`: contribution `-0.004488`
- `lag_12__T3__flash_duration`: contribution `-0.004423`
- `lag_02__CT_flash_duration_sum`: contribution `-0.002735`

### tick `64123`, seconds `16.50`, LSTM delta `-0.0605`

Top all feature movements:
- `lag_11__CT_place_ELECTRICALBOX`: contribution `+0.018613`
- `lag_08__CT_place_ELECTRICALBOX`: contribution `-0.017611`
- `lag_13__CT_place_ELECTRICALBOX`: contribution `-0.015011`
- `lag_13__T_place_DUMPSTER`: contribution `+0.009042`
- `lag_03__CT1__flash_duration`: contribution `-0.005189`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.005189`
- `lag_01__CT1__flash_duration`: contribution `+0.004034`
- `lag_13__T3__flash_duration`: contribution `-0.002106`
- `lag_07__CT2__flash_duration`: contribution `-0.001787`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001617`

### tick `64059`, seconds `15.50`, LSTM delta `-0.0439`

Top all feature movements:
- `lag_11__CT_place_ELECTRICALBOX`: contribution `-0.018613`
- `lag_12__CT_place_ELECTRICALBOX`: contribution `-0.013170`
- `lag_01__CT1__flash_duration`: contribution `-0.004034`
- `lag_15__T_place_TSTAIRS`: contribution `-0.003069`
- `lag_05__T2__duck_amount`: contribution `+0.002601`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `-0.004034`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.001657`
- `lag_11__T3__flash_duration`: contribution `-0.001613`
- `lag_01__CT4__flash_duration`: contribution `-0.001251`
- `lag_01__CT5__flash_duration`: contribution `-0.001063`

### tick `63323`, seconds `4.00`, LSTM delta `-0.0296`

Top all feature movements:
- `lag_03__CT_place_ENTRANCE`: contribution `-0.004942`
- `lag_02__CT_place_ENTRANCE`: contribution `-0.004103`
- `lag_04__CT_place_ENTRANCE`: contribution `-0.004041`
- `lag_05__CT_place_ENTRANCE`: contribution `-0.003013`
- `lag_00__CT_place_CTSPAWN`: contribution `-0.000874`

Top utility-only movements:
- `lag_08__CT3__utility_total`: contribution `-0.000710`
- `lag_08__CT3__molly`: contribution `-0.000652`
- `lag_08__CT5__smoke`: contribution `+0.000571`
- `lag_08__T5__utility_total`: contribution `-0.000487`
- `lag_08__T2__flash`: contribution `+0.000427`
