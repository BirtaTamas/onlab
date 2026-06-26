# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `34266`, seconds `45.50`, LSTM `0.7360`, delta `+0.1506`
- tick `34330`, seconds `46.50`, LSTM `0.9183`, delta `+0.1265`
- tick `34298`, seconds `46.00`, LSTM `0.7919`, delta `+0.0559`
- tick `34202`, seconds `44.50`, LSTM `0.5450`, delta `+0.0478`
- tick `34234`, seconds `45.00`, LSTM `0.5854`, delta `+0.0405`
- tick `34170`, seconds `44.00`, LSTM `0.4972`, delta `-0.0313`
- tick `33562`, seconds `34.50`, LSTM `0.5850`, delta `-0.0268`
- tick `33146`, seconds `28.00`, LSTM `0.6403`, delta `+0.0255`
- tick `34138`, seconds `43.50`, LSTM `0.5284`, delta `-0.0206`
- tick `33498`, seconds `33.50`, LSTM `0.6052`, delta `-0.0194`

## Top 15 local ridge features

- `lag_02__T_shots_fired_sum`: coefficient `-0.002218`, |coef| `0.002218`
- `lag_02__T3__shots_fired`: coefficient `-0.001782`, |coef| `0.001782`
- `lag_05__T_shots_fired_sum`: coefficient `0.001603`, |coef| `0.001603`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001458`, |coef| `0.001458`
- `lag_06__T_shots_fired_sum`: coefficient `0.001413`, |coef| `0.001413`
- `lag_05__T3__shots_fired`: coefficient `0.001278`, |coef| `0.001278`
- `lag_00__T3__shots_fired`: coefficient `-0.001193`, |coef| `0.001193`
- `lag_06__T3__shots_fired`: coefficient `0.001124`, |coef| `0.001124`
- `lag_07__T_place_SHORTSTAIRS`: coefficient `0.001063`, |coef| `0.001063`
- `lag_00__CT_kills_last_3s`: coefficient `0.001059`, |coef| `0.001059`
- `lag_00__CT_place_HOLE`: coefficient `0.001045`, |coef| `0.001045`
- `lag_01__T_shots_fired_sum`: coefficient `-0.000926`, |coef| `0.000926`
- `lag_09__T_place_SHORTSTAIRS`: coefficient `0.000906`, |coef| `0.000906`
- `lag_07__T_shots_fired_sum`: coefficient `0.000904`, |coef| `0.000904`
- `lag_00__kill_diff_last_3s`: coefficient `0.000883`, |coef| `0.000883`

## Top 10 utility ridge features

- `lag_02__CT_active_infernos`: coefficient `-0.000799` (lowers CT win probability)
- `lag_01__CT1__flash_duration`: coefficient `0.000751` (raises CT win probability)
- `lag_00__T4__smoke`: coefficient `-0.000694` (lowers CT win probability)
- `lag_04__CT_active_infernos`: coefficient `-0.000693` (lowers CT win probability)
- `lag_02__CT_A_site_active_infernos`: coefficient `-0.000681` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.000665` (lowers CT win probability)
- `lag_15__CT_active_smokes`: coefficient `0.000615` (raises CT win probability)
- `lag_01__CT_flash_duration_sum`: coefficient `0.000577` (raises CT win probability)
- `lag_10__T_smokes_last_5s`: coefficient `0.000509` (raises CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `0.000491` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__T_shots_fired_sum`: coefficient `-0.002218` (lowers CT win probability)
- `lag_02__T3__shots_fired`: coefficient `-0.001782` (lowers CT win probability)
- `lag_05__T_shots_fired_sum`: coefficient `0.001603` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001458` (lowers CT win probability)
- `lag_06__T_shots_fired_sum`: coefficient `0.001413` (raises CT win probability)
- `lag_05__T3__shots_fired`: coefficient `0.001278` (raises CT win probability)
- `lag_00__T3__shots_fired`: coefficient `-0.001193` (lowers CT win probability)
- `lag_06__T3__shots_fired`: coefficient `0.001124` (raises CT win probability)
- `lag_07__T_place_SHORTSTAIRS`: coefficient `0.001063` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001059` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `34266`, seconds `45.50`, LSTM delta `+0.1506`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `+0.031594`
- `lag_02__T3__shots_fired`: contribution `+0.020504`
- `lag_05__T_shots_fired_sum`: contribution `+0.006008`
- `lag_07__T_place_SHORTSTAIRS`: contribution `+0.004466`
- `lag_06__T_shots_fired_sum`: contribution `+0.004237`

Top utility-only movements:
- `lag_02__CT_active_infernos`: contribution `+0.003680`
- `lag_02__CT_A_site_active_infernos`: contribution `+0.002405`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.002284`

### tick `34330`, seconds `46.50`, LSTM delta `+0.1265`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `+0.006008`
- `lag_06__T_shots_fired_sum`: contribution `+0.005296`
- `lag_04__T_flashed_players`: contribution `+0.003920`
- `lag_05__T3__shots_fired`: contribution `+0.003868`
- `lag_09__T_place_SHORTSTAIRS`: contribution `+0.003808`

Top utility-only movements:
- `lag_01__CT1__flash_duration`: contribution `+0.003802`
- `lag_04__CT_active_infernos`: contribution `+0.003192`
- `lag_01__CT_flash_duration_sum`: contribution `+0.002029`
- `lag_04__CT_A_site_active_infernos`: contribution `+0.001707`

### tick `34298`, seconds `46.00`, LSTM delta `+0.0559`

Top all feature movements:
- `lag_05__T_shots_fired_sum`: contribution `+0.006008`
- `lag_06__T_shots_fired_sum`: contribution `+0.005296`
- `lag_05__T3__shots_fired`: contribution `+0.003868`
- `lag_06__T3__shots_fired`: contribution `+0.003403`
- `lag_08__T_place_SHORTSTAIRS`: contribution `+0.002850`

Top utility-only movements:
- `lag_00__CT1__flash_duration`: contribution `+0.002488`
- `lag_03__CT_active_infernos`: contribution `+0.001223`
- `lag_00__CT_flash_duration_sum`: contribution `+0.001170`

### tick `34202`, seconds `44.50`, LSTM delta `+0.0478`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.020766`
- `lag_00__T3__shots_fired`: contribution `+0.013732`
- `lag_02__T_shots_fired_sum`: contribution `-0.008314`
- `lag_02__T3__shots_fired`: contribution `-0.005396`
- `lag_01__T_shots_fired_sum`: contribution `-0.003473`

Top utility-only movements:
- `lag_14__CT_active_infernos`: contribution `+0.001747`
- `lag_14__CT_A_site_active_infernos`: contribution `+0.001600`
- `lag_14__CT_B_site_active_infernos`: contribution `+0.001512`
- `lag_00__CT_active_infernos`: contribution `+0.001369`
- `lag_00__CT_A_site_active_infernos`: contribution `+0.000915`

### tick `34234`, seconds `45.00`, LSTM delta `+0.0405`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.013197`
- `lag_01__T3__shots_fired`: contribution `+0.008688`
- `lag_02__T_shots_fired_sum`: contribution `-0.008314`
- `lag_02__T3__shots_fired`: contribution `-0.005396`
- `lag_05__T_shots_fired_sum`: contribution `+0.004807`

Top utility-only movements:
- `lag_01__CT_active_infernos`: contribution `+0.001763`
- `lag_15__CT_active_infernos`: contribution `+0.001758`
- `lag_15__CT_A_site_active_infernos`: contribution `+0.001593`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001504`
