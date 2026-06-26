# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `18`

## Largest probability jumps

- tick `156526`, seconds `54.50`, LSTM `0.0538`, delta `-0.1603`
- tick `156430`, seconds `53.00`, LSTM `0.3717`, delta `-0.1582`
- tick `156494`, seconds `54.00`, LSTM `0.2141`, delta `-0.1452`
- tick `154510`, seconds `23.00`, LSTM `0.5669`, delta `+0.0463`
- tick `155214`, seconds `34.00`, LSTM `0.5442`, delta `-0.0339`
- tick `153742`, seconds `11.00`, LSTM `0.5145`, delta `+0.0291`
- tick `156558`, seconds `55.00`, LSTM `0.0261`, delta `-0.0276`
- tick `154734`, seconds `26.50`, LSTM `0.5777`, delta `+0.0231`
- tick `154638`, seconds `25.00`, LSTM `0.5385`, delta `-0.0210`
- tick `154766`, seconds `27.00`, LSTM `0.5581`, delta `-0.0196`

## Top 15 local ridge features

- `lag_14__T_place_DUMPSTER`: coefficient `-0.001815`, |coef| `0.001815`
- `lag_13__T_place_DUMPSTER`: coefficient `-0.001770`, |coef| `0.001770`
- `lag_11__T_place_DUMPSTER`: coefficient `-0.001679`, |coef| `0.001679`
- `lag_00__T_kills_last_3s`: coefficient `-0.001587`, |coef| `0.001587`
- `lag_00__CT_place_LONGDOG`: coefficient `0.001462`, |coef| `0.001462`
- `lag_00__T_damage_last_5s`: coefficient `-0.001397`, |coef| `0.001397`
- `lag_15__T_place_DUMPSTER`: coefficient `-0.001293`, |coef| `0.001293`
- `lag_00__damage_diff_last_5s`: coefficient `0.001237`, |coef| `0.001237`
- `lag_07__T_place_DUMPSTER`: coefficient `-0.001223`, |coef| `0.001223`
- `lag_00__kill_diff_last_3s`: coefficient `0.001210`, |coef| `0.001210`
- `lag_08__T_place_ALLEY`: coefficient `0.001148`, |coef| `0.001148`
- `lag_15__T5__duck_amount`: coefficient `-0.001144`, |coef| `0.001144`
- `lag_02__CT_place_LONGDOG`: coefficient `0.001144`, |coef| `0.001144`
- `lag_12__T_place_DUMPSTER`: coefficient `-0.001142`, |coef| `0.001142`
- `lag_00__T_place_TSTAIRS`: coefficient `-0.001113`, |coef| `0.001113`

## Top 10 utility ridge features

- `lag_00__CT2__molly`: coefficient `0.000986` (raises CT win probability)
- `lag_01__CT2__molly`: coefficient `0.000791` (raises CT win probability)
- `lag_07__CT_A_site_active_smokes`: coefficient `0.000790` (raises CT win probability)
- `lag_03__T1__flash_duration`: coefficient `0.000632` (raises CT win probability)
- `lag_09__CT_A_site_active_smokes`: coefficient `0.000622` (raises CT win probability)
- `lag_10__CT_A_site_active_smokes`: coefficient `0.000622` (raises CT win probability)
- `lag_06__CT_A_site_active_smokes`: coefficient `0.000595` (raises CT win probability)
- `lag_07__CT_active_smokes`: coefficient `0.000594` (raises CT win probability)
- `lag_06__T_A_site_active_smokes`: coefficient `0.000572` (raises CT win probability)
- `lag_06__active_smokes_total`: coefficient `0.000567` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_DUMPSTER`: coefficient `-0.001815` (lowers CT win probability)
- `lag_13__T_place_DUMPSTER`: coefficient `-0.001770` (lowers CT win probability)
- `lag_11__T_place_DUMPSTER`: coefficient `-0.001679` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001587` (lowers CT win probability)
- `lag_00__CT_place_LONGDOG`: coefficient `0.001462` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001397` (lowers CT win probability)
- `lag_15__T_place_DUMPSTER`: coefficient `-0.001293` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001237` (raises CT win probability)
- `lag_07__T_place_DUMPSTER`: coefficient `-0.001223` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001210` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `156526`, seconds `54.50`, LSTM delta `-0.1603`

Top all feature movements:
- `lag_14__T_place_DUMPSTER`: contribution `-0.016507`
- `lag_02__T_place_DUMPSTER`: contribution `-0.007487`
- `lag_08__T_place_DUMPSTER`: contribution `-0.006786`
- `lag_03__CT_place_LONGDOG`: contribution `-0.006610`
- `lag_00__T_kills_last_3s`: contribution `-0.005028`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `156430`, seconds `53.00`, LSTM delta `-0.1582`

Top all feature movements:
- `lag_11__T_place_DUMPSTER`: contribution `-0.015268`
- `lag_00__CT_place_LONGDOG`: contribution `-0.009536`
- `lag_05__T_place_DUMPSTER`: contribution `-0.005847`
- `lag_00__T_kills_last_3s`: contribution `-0.005028`
- `lag_05__T_place_ALLEY`: contribution `-0.004410`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `156494`, seconds `54.00`, LSTM delta `-0.1452`

Top all feature movements:
- `lag_13__T_place_DUMPSTER`: contribution `-0.016092`
- `lag_07__T_place_DUMPSTER`: contribution `-0.011117`
- `lag_01__T_place_DUMPSTER`: contribution `-0.009729`
- `lag_02__CT_place_LONGDOG`: contribution `-0.007461`
- `lag_08__T_place_DUMPSTER`: contribution `+0.006786`

Top utility-only movements:
- `lag_00__CT2__molly`: contribution `-0.002432`

### tick `154510`, seconds `23.00`, LSTM delta `+0.0463`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.005028`
- `lag_03__CT_place_ELECTRICALBOX`: contribution `+0.004702`
- `lag_03__T1__flash_duration`: contribution `+0.004647`
- `lag_02__CT_place_ELECTRICALBOX`: contribution `+0.003237`
- `lag_03__T4__flash_duration`: contribution `+0.003096`

Top utility-only movements:
- `lag_03__T1__flash_duration`: contribution `+0.004647`
- `lag_03__T4__flash_duration`: contribution `+0.003096`
- `lag_03__T_flash_duration_sum`: contribution `+0.002555`

### tick `155214`, seconds `34.00`, LSTM delta `-0.0339`

Top all feature movements:
- `lag_11__T1__flash_duration`: contribution `-0.002943`
- `lag_00__CT3__flash_duration`: contribution `-0.002547`
- `lag_02__CT3__duck_amount`: contribution `-0.002152`
- `lag_00__T4__flash_duration`: contribution `-0.001876`
- `lag_05__T4__duck_amount`: contribution `-0.001863`

Top utility-only movements:
- `lag_11__T1__flash_duration`: contribution `-0.002943`
- `lag_00__CT3__flash_duration`: contribution `-0.002547`
- `lag_00__T4__flash_duration`: contribution `-0.001876`
- `lag_09__T1__flash_duration`: contribution `+0.001611`
- `lag_12__T4__flash_duration`: contribution `-0.001401`
