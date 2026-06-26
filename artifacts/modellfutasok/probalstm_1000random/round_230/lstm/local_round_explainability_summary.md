# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-natus-vincere-bo3-FVT9m_t7tlOrOuiYTIheUW/the-mongolz-vs-natus-vincere-m2-inferno.csv`
- round_num: `10`

## Largest probability jumps

- tick `79249`, seconds `13.50`, LSTM `0.8659`, delta `+0.0881`
- tick `79217`, seconds `13.00`, LSTM `0.7778`, delta `+0.0659`
- tick `79185`, seconds `12.50`, LSTM `0.7120`, delta `+0.0508`
- tick `84945`, seconds `102.50`, LSTM `0.9502`, delta `+0.0502`
- tick `79281`, seconds `14.00`, LSTM `0.9039`, delta `+0.0380`
- tick `84753`, seconds `99.50`, LSTM `0.9069`, delta `+0.0355`
- tick `79857`, seconds `23.00`, LSTM `0.8677`, delta `-0.0335`
- tick `79921`, seconds `24.00`, LSTM `0.8912`, delta `+0.0277`
- tick `78929`, seconds `8.50`, LSTM `0.6715`, delta `+0.0270`
- tick `79345`, seconds `15.00`, LSTM `0.8922`, delta `-0.0253`

## Top 15 local ridge features

- `lag_00__CT5__duck_amount`: coefficient `0.001060`, |coef| `0.001060`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001022`, |coef| `0.001022`
- `lag_00__CT2__shots_fired`: coefficient `0.000821`, |coef| `0.000821`
- `lag_00__CT_duck_amount_mean`: coefficient `0.000646`, |coef| `0.000646`
- `lag_04__CT_place_LIBRARY`: coefficient `0.000645`, |coef| `0.000645`
- `lag_11__T_place_QUAD`: coefficient `0.000642`, |coef| `0.000642`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000597`, |coef| `0.000597`
- `lag_00__T_place_QUAD`: coefficient `0.000592`, |coef| `0.000592`
- `lag_05__CT_place_LIBRARY`: coefficient `0.000583`, |coef| `0.000583`
- `lag_00__CT_kills_last_3s`: coefficient `0.000570`, |coef| `0.000570`
- `lag_08__CT_place_LIBRARY`: coefficient `0.000569`, |coef| `0.000569`
- `lag_07__CT_place_BANANA`: coefficient `0.000564`, |coef| `0.000564`
- `lag_13__T_place_LOWERMID`: coefficient `-0.000552`, |coef| `0.000552`
- `lag_06__T_place_QUAD`: coefficient `0.000537`, |coef| `0.000537`
- `lag_06__T_place_BALCONY`: coefficient `0.000518`, |coef| `0.000518`

## Top 10 utility ridge features

- `lag_05__T4__flash_duration`: coefficient `0.000365` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000349` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.000336` (lowers CT win probability)
- `lag_04__T2__flash_duration`: coefficient `-0.000330` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000329` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000321` (lowers CT win probability)
- `lag_05__T1__flash_duration`: coefficient `0.000315` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.000293` (raises CT win probability)
- `lag_00__CT3__flash`: coefficient `-0.000286` (lowers CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000284` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT5__duck_amount`: coefficient `0.001060` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001022` (raises CT win probability)
- `lag_00__CT2__shots_fired`: coefficient `0.000821` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.000646` (raises CT win probability)
- `lag_04__CT_place_LIBRARY`: coefficient `0.000645` (raises CT win probability)
- `lag_11__T_place_QUAD`: coefficient `0.000642` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000597` (raises CT win probability)
- `lag_00__T_place_QUAD`: coefficient `0.000592` (raises CT win probability)
- `lag_05__CT_place_LIBRARY`: coefficient `0.000583` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000570` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `79249`, seconds `13.50`, LSTM delta `+0.0881`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `-0.004970`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004566`
- `lag_13__T_place_LOWERMID`: contribution `+0.003673`
- `lag_13__T_place_TRAMP`: contribution `+0.002611`
- `lag_05__T4__flash_duration`: contribution `+0.002144`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.002144`
- `lag_05__T1__flash_duration`: contribution `+0.001576`
- `lag_01__T4__flash_duration`: contribution `+0.001399`

### tick `79217`, seconds `13.00`, LSTM delta `+0.0659`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007811`
- `lag_01__CT_shots_fired_sum`: contribution `+0.004151`
- `lag_00__CT2__shots_fired`: contribution `+0.002039`
- `lag_12__T_place_TRAMP`: contribution `+0.002030`
- `lag_13__T_place_LOWERMID`: contribution `+0.001836`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.001725`

### tick `79185`, seconds `12.50`, LSTM delta `+0.0508`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.007101`
- `lag_11__T_place_TRAMP`: contribution `+0.002678`
- `lag_00__CT2__shots_fired`: contribution `+0.002039`
- `lag_13__T_place_LOWERMID`: contribution `+0.001836`
- `lag_09__CT_place_RUINS`: contribution `+0.001804`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `84945`, seconds `102.50`, LSTM delta `+0.0502`

Top all feature movements:
- `lag_06__T_place_QUAD`: contribution `+0.012932`
- `lag_06__T_place_BALCONY`: contribution `+0.007124`
- `lag_00__T_place_BALCONY`: contribution `-0.006911`
- `lag_01__T_place_QUAD`: contribution `+0.006874`
- `lag_00__CT5__duck_amount`: contribution `+0.003117`

Top utility-only movements:
- `lag_12__CT_B_site_active_infernos`: contribution `+0.000709`

### tick `79281`, seconds `14.00`, LSTM delta `+0.0380`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004260`
- `lag_14__T_place_LOWERMID`: contribution `+0.002928`
- `lag_01__CT_shots_fired_sum`: contribution `-0.002906`
- `lag_03__CT_shots_fired_sum`: contribution `+0.002526`
- `lag_00__CT2__shots_fired`: contribution `+0.002039`

Top utility-only movements:
- `lag_02__T4__flash_duration`: contribution `+0.000693`
