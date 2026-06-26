# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `4`

## Largest probability jumps

- tick `25447`, seconds `34.50`, LSTM `0.8927`, delta `+0.1550`
- tick `24135`, seconds `14.00`, LSTM `0.6381`, delta `-0.1296`
- tick `24263`, seconds `16.00`, LSTM `0.7306`, delta `+0.1231`
- tick `26279`, seconds `47.50`, LSTM `0.9477`, delta `+0.0760`
- tick `23783`, seconds `8.50`, LSTM `0.6802`, delta `-0.0449`
- tick `24999`, seconds `27.50`, LSTM `0.7797`, delta `+0.0434`
- tick `24903`, seconds `26.00`, LSTM `0.7049`, delta `-0.0411`
- tick `24967`, seconds `27.00`, LSTM `0.7363`, delta `+0.0387`
- tick `25319`, seconds `32.50`, LSTM `0.7807`, delta `+0.0339`
- tick `24711`, seconds `23.00`, LSTM `0.7271`, delta `-0.0327`

## Top 15 local ridge features

- `lag_15__CT_place_UPPERTUNNEL`: coefficient `0.001649`, |coef| `0.001649`
- `lag_11__CT_place_HOLE`: coefficient `0.001324`, |coef| `0.001324`
- `lag_04__CT_place_HOLE`: coefficient `0.001244`, |coef| `0.001244`
- `lag_02__T_place_TUNNELSTAIRS`: coefficient `-0.001153`, |coef| `0.001153`
- `lag_11__CT_place_UNDERA`: coefficient `0.001144`, |coef| `0.001144`
- `lag_00__kill_diff_last_3s`: coefficient `0.001104`, |coef| `0.001104`
- `lag_15__T_flashes_last_5s`: coefficient `0.001069`, |coef| `0.001069`
- `lag_00__CT_kills_last_3s`: coefficient `0.001049`, |coef| `0.001049`
- `lag_15__CT_place_HOLE`: coefficient `-0.001017`, |coef| `0.001017`
- `lag_11__T_flashes_last_5s`: coefficient `-0.001005`, |coef| `0.001005`
- `lag_02__CT_place_HOLE`: coefficient `-0.000965`, |coef| `0.000965`
- `lag_13__CT_place_HOLE`: coefficient `-0.000921`, |coef| `0.000921`
- `lag_00__damage_diff_last_5s`: coefficient `0.000918`, |coef| `0.000918`
- `lag_08__CT2__duck_amount`: coefficient `-0.000888`, |coef| `0.000888`
- `lag_00__CT2__duck_amount`: coefficient `0.000881`, |coef| `0.000881`

## Top 10 utility ridge features

- `lag_15__T_flashes_last_5s`: coefficient `0.001069` (raises CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `-0.001005` (lowers CT win probability)
- `lag_12__CT_B_site_active_infernos`: coefficient `-0.000782` (lowers CT win probability)
- `lag_05__T_flashes_last_5s`: coefficient `-0.000765` (lowers CT win probability)
- `lag_03__CT5__flash_duration`: coefficient `0.000700` (raises CT win probability)
- `lag_00__T_flashes_last_5s`: coefficient `-0.000695` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.000629` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `-0.000613` (lowers CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.000584` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000580` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_UPPERTUNNEL`: coefficient `0.001649` (raises CT win probability)
- `lag_11__CT_place_HOLE`: coefficient `0.001324` (raises CT win probability)
- `lag_04__CT_place_HOLE`: coefficient `0.001244` (raises CT win probability)
- `lag_02__T_place_TUNNELSTAIRS`: coefficient `-0.001153` (lowers CT win probability)
- `lag_11__CT_place_UNDERA`: coefficient `0.001144` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001104` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001049` (raises CT win probability)
- `lag_15__CT_place_HOLE`: coefficient `-0.001017` (lowers CT win probability)
- `lag_02__CT_place_HOLE`: coefficient `-0.000965` (lowers CT win probability)
- `lag_13__CT_place_HOLE`: coefficient `-0.000921` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `25447`, seconds `34.50`, LSTM delta `+0.1550`

Top all feature movements:
- `lag_04__CT_place_HOLE`: contribution `+0.013891`
- `lag_15__CT_place_UPPERTUNNEL`: contribution `+0.012651`
- `lag_01__CT_place_HOLE`: contribution `+0.008406`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `+0.006675`
- `lag_00__CT3__is_scoped`: contribution `+0.003795`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `+0.001984`

### tick `24135`, seconds `14.00`, LSTM delta `-0.1296`

Top all feature movements:
- `lag_11__CT_place_HOLE`: contribution `-0.014783`
- `lag_13__CT_place_HOLE`: contribution `-0.010280`
- `lag_11__T_flashes_last_5s`: contribution `-0.009109`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `-0.008048`
- `lag_04__T_place_TUNNELSTAIRS`: contribution `-0.005160`

Top utility-only movements:
- `lag_11__T_flashes_last_5s`: contribution `-0.009109`
- `lag_01__T_flashes_last_5s`: contribution `-0.004058`
- `lag_03__CT5__flash_duration`: contribution `-0.002723`
- `lag_12__CT_B_site_active_infernos`: contribution `-0.002686`
- `lag_03__T5__flash_duration`: contribution `-0.001511`

### tick `24263`, seconds `16.00`, LSTM delta `+0.1231`

Top all feature movements:
- `lag_15__CT_place_HOLE`: contribution `+0.011352`
- `lag_15__T_flashes_last_5s`: contribution `+0.009682`
- `lag_02__T_place_TUNNELSTAIRS`: contribution `+0.008048`
- `lag_05__T_flashes_last_5s`: contribution `+0.006933`
- `lag_08__T_place_TUNNELSTAIRS`: contribution `+0.003981`

Top utility-only movements:
- `lag_15__T_flashes_last_5s`: contribution `+0.009682`
- `lag_05__T_flashes_last_5s`: contribution `+0.006933`
- `lag_07__T5__flash_duration`: contribution `+0.001687`
- `lag_13__CT5__flash_duration`: contribution `+0.001588`

### tick `26279`, seconds `47.50`, LSTM delta `+0.0760`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004156`
- `lag_00__T_shots_fired_sum`: contribution `+0.003047`
- `lag_00__CT_kills_last_3s`: contribution `+0.003028`
- `lag_09__CT_place_BDOORS`: contribution `+0.002998`
- `lag_04__CT_place_EXTENDEDA`: contribution `+0.002731`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23783`, seconds `8.50`, LSTM delta `-0.0449`

Top all feature movements:
- `lag_02__CT_place_HOLE`: contribution `-0.010775`
- `lag_00__CT_place_HOLE`: contribution `-0.007044`
- `lag_00__T_flashes_last_5s`: contribution `-0.006300`
- `lag_08__CT3__duck_amount`: contribution `+0.002681`
- `lag_07__CT3__is_scoped`: contribution `-0.002524`

Top utility-only movements:
- `lag_00__T_flashes_last_5s`: contribution `-0.006300`
- `lag_01__CT_B_site_active_infernos`: contribution `+0.001339`
- `lag_08__CT4__smoke`: contribution `-0.000905`
