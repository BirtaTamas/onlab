# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `1`

## Largest probability jumps

- tick `8887`, seconds `56.00`, LSTM `0.1286`, delta `-0.3154`
- tick `8855`, seconds `55.50`, LSTM `0.4440`, delta `+0.2826`
- tick `8695`, seconds `53.00`, LSTM `0.2514`, delta `-0.2601`
- tick `8919`, seconds `56.50`, LSTM `0.0311`, delta `-0.0975`
- tick `8727`, seconds `53.50`, LSTM `0.1768`, delta `-0.0746`
- tick `9367`, seconds `63.50`, LSTM `0.0207`, delta `-0.0302`
- tick `9047`, seconds `58.50`, LSTM `0.0298`, delta `+0.0173`
- tick `8631`, seconds `52.00`, LSTM `0.5098`, delta `+0.0120`
- tick `8759`, seconds `54.00`, LSTM `0.1653`, delta `-0.0114`
- tick `8567`, seconds `51.00`, LSTM `0.4968`, delta `-0.0114`

## Top 15 local ridge features

- `lag_09__T_place_SQUEAKY`: coefficient `-0.003059`, |coef| `0.003059`
- `lag_10__T_place_SQUEAKY`: coefficient `0.002793`, |coef| `0.002793`
- `lag_04__T_place_SQUEAKY`: coefficient `0.002178`, |coef| `0.002178`
- `lag_01__T_bomb_zone_count`: coefficient `0.001945`, |coef| `0.001945`
- `lag_01__T_place_HUT`: coefficient `-0.001742`, |coef| `0.001742`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001687`, |coef| `0.001687`
- `lag_11__CT3__flash_duration`: coefficient `0.001680`, |coef| `0.001680`
- `lag_00__damage_diff_last_5s`: coefficient `0.001679`, |coef| `0.001679`
- `lag_05__CT5__flash_duration`: coefficient `-0.001670`, |coef| `0.001670`
- `lag_06__T_place_SQUEAKY`: coefficient `-0.001610`, |coef| `0.001610`
- `lag_05__CT_place_SECRET`: coefficient `0.001574`, |coef| `0.001574`
- `lag_11__CT5__flash_duration`: coefficient `-0.001548`, |coef| `0.001548`
- `lag_12__CT3__flash_duration`: coefficient `-0.001514`, |coef| `0.001514`
- `lag_04__CT_place_SECRET`: coefficient `-0.001405`, |coef| `0.001405`
- `lag_00__T_damage_last_5s`: coefficient `-0.001384`, |coef| `0.001384`

## Top 10 utility ridge features

- `lag_11__CT3__flash_duration`: coefficient `0.001680` (raises CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `-0.001670` (lowers CT win probability)
- `lag_11__CT5__flash_duration`: coefficient `-0.001548` (lowers CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `-0.001514` (lowers CT win probability)
- `lag_06__CT3__flash_duration`: coefficient `-0.001341` (lowers CT win probability)
- `lag_10__CT5__flash_duration`: coefficient `0.001127` (raises CT win probability)
- `lag_05__CT_flash_duration_sum`: coefficient `-0.001086` (lowers CT win probability)
- `lag_12__CT_flash_duration_sum`: coefficient `-0.000996` (lowers CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `-0.000946` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.000910` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_09__T_place_SQUEAKY`: coefficient `-0.003059` (lowers CT win probability)
- `lag_10__T_place_SQUEAKY`: coefficient `0.002793` (raises CT win probability)
- `lag_04__T_place_SQUEAKY`: coefficient `0.002178` (raises CT win probability)
- `lag_01__T_bomb_zone_count`: coefficient `0.001945` (raises CT win probability)
- `lag_01__T_place_HUT`: coefficient `-0.001742` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001687` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001679` (raises CT win probability)
- `lag_06__T_place_SQUEAKY`: coefficient `-0.001610` (lowers CT win probability)
- `lag_05__CT_place_SECRET`: coefficient `0.001574` (raises CT win probability)
- `lag_04__CT_place_SECRET`: coefficient `-0.001405` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `8887`, seconds `56.00`, LSTM delta `-0.3154`

Top all feature movements:
- `lag_10__T_place_SQUEAKY`: contribution `-0.034785`
- `lag_01__T_place_HUT`: contribution `-0.016240`
- `lag_05__CT_place_SECRET`: contribution `-0.016202`
- `lag_12__CT3__flash_duration`: contribution `-0.011708`
- `lag_01__T_bomb_zone_count`: contribution `-0.011322`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `-0.011708`
- `lag_11__CT5__flash_duration`: contribution `-0.010722`
- `lag_06__CT3__flash_duration`: contribution `+0.005150`

### tick `8855`, seconds `55.50`, LSTM delta `+0.2826`

Top all feature movements:
- `lag_09__T_place_SQUEAKY`: contribution `+0.038085`
- `lag_10__T_place_SQUEAKY`: contribution `+0.017393`
- `lag_04__CT_place_SECRET`: contribution `+0.014458`
- `lag_11__CT3__flash_duration`: contribution `+0.012987`
- `lag_01__T_bomb_zone_count`: contribution `+0.011322`

Top utility-only movements:
- `lag_11__CT3__flash_duration`: contribution `+0.012987`
- `lag_10__CT5__flash_duration`: contribution `+0.007807`

### tick `8695`, seconds `53.00`, LSTM delta `-0.2601`

Top all feature movements:
- `lag_04__T_place_SQUEAKY`: contribution `-0.027120`
- `lag_09__T_place_SQUEAKY`: contribution `-0.019043`
- `lag_05__CT5__flash_duration`: contribution `-0.011565`
- `lag_06__CT3__flash_duration`: contribution `-0.010366`
- `lag_01__T_place_SILO`: contribution `-0.009235`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.011565`
- `lag_06__CT3__flash_duration`: contribution `-0.010366`

### tick `8919`, seconds `56.50`, LSTM delta `-0.0975`

Top all feature movements:
- `lag_09__T_place_SQUEAKY`: contribution `+0.019043`
- `lag_11__T_place_SQUEAKY`: contribution `+0.009368`
- `lag_08__T_place_SQUEAKY`: contribution `-0.007687`
- `lag_02__CT_place_ADMIN`: contribution `-0.006918`
- `lag_02__T_place_HUT`: contribution `-0.005794`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `-0.005547`
- `lag_12__CT5__flash_duration`: contribution `-0.004867`
- `lag_01__CT5__flash_duration`: contribution `-0.002720`

### tick `8727`, seconds `53.50`, LSTM delta `-0.0746`

Top all feature movements:
- `lag_10__T_place_SQUEAKY`: contribution `+0.017393`
- `lag_06__T_place_SQUEAKY`: contribution `-0.010025`
- `lag_05__T_place_SQUEAKY`: contribution `+0.008123`
- `lag_02__T_place_ROOF`: contribution `-0.007714`
- `lag_02__T_place_SQUEAKY`: contribution `-0.007042`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `-0.006551`
- `lag_06__CT_flash_duration_sum`: contribution `-0.002182`
- `lag_06__CT3__flash_duration`: contribution `+0.001872`
- `lag_01__CT3__flash_duration`: contribution `-0.001747`
