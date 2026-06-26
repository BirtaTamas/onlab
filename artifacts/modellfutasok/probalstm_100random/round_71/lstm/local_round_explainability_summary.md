# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `2`

## Largest probability jumps

- tick `21321`, seconds `96.50`, LSTM `0.4572`, delta `-0.3141`
- tick `21161`, seconds `94.00`, LSTM `0.8473`, delta `+0.2755`
- tick `21193`, seconds `94.50`, LSTM `0.6206`, delta `-0.2267`
- tick `21961`, seconds `106.50`, LSTM `0.3344`, delta `+0.2240`
- tick `22281`, seconds `111.50`, LSTM `0.5596`, delta `+0.1780`
- tick `16137`, seconds `15.50`, LSTM `0.5657`, delta `-0.1688`
- tick `19177`, seconds `63.00`, LSTM `0.7257`, delta `+0.1682`
- tick `20201`, seconds `79.00`, LSTM `0.8578`, delta `+0.1548`
- tick `21097`, seconds `93.00`, LSTM `0.5990`, delta `-0.1425`
- tick `21353`, seconds `97.00`, LSTM `0.3269`, delta `-0.1303`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005852`, |coef| `0.005852`
- `lag_00__kill_diff_last_3s`: coefficient `0.005793`, |coef| `0.005793`
- `lag_00__CT_kills_last_3s`: coefficient `0.004702`, |coef| `0.004702`
- `lag_00__CT_duck_amount_mean`: coefficient `0.004604`, |coef| `0.004604`
- `lag_00__CT_shots_fired_sum`: coefficient `0.004001`, |coef| `0.004001`
- `lag_07__T_bomb_zone_count`: coefficient `-0.003927`, |coef| `0.003927`
- `lag_13__T_place_VENTS`: coefficient `0.003116`, |coef| `0.003116`
- `lag_08__CT_place_MINI`: coefficient `0.003098`, |coef| `0.003098`
- `lag_00__T_place_SILO`: coefficient `-0.002791`, |coef| `0.002791`
- `lag_00__CT_velocity_mean`: coefficient `-0.002745`, |coef| `0.002745`
- `lag_10__CT_defusing_count`: coefficient `-0.002707`, |coef| `0.002707`
- `lag_02__CT_place_GARAGE`: coefficient `-0.002635`, |coef| `0.002635`
- `lag_03__T_place_MINI`: coefficient `-0.002586`, |coef| `0.002586`
- `lag_00__CT4__duck_amount`: coefficient `0.002546`, |coef| `0.002546`
- `lag_10__damage_diff_last_5s`: coefficient `0.002475`, |coef| `0.002475`

## Top 10 utility ridge features

- `lag_00__CT3__flash`: coefficient `0.001195` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.000994` (lowers CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `-0.000981` (lowers CT win probability)
- `lag_01__T_flash_alpha_mean`: coefficient `-0.000965` (lowers CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `-0.000951` (lowers CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.000945` (lowers CT win probability)
- `lag_04__T4__flash_duration`: coefficient `-0.000874` (lowers CT win probability)
- `lag_04__CT_B_site_active_smokes`: coefficient `-0.000843` (lowers CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `0.000841` (raises CT win probability)
- `lag_14__CT_flashes_last_5s`: coefficient `-0.000833` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.005852` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005793` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004702` (raises CT win probability)
- `lag_00__CT_duck_amount_mean`: coefficient `0.004604` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.004001` (raises CT win probability)
- `lag_07__T_bomb_zone_count`: coefficient `-0.003927` (lowers CT win probability)
- `lag_13__T_place_VENTS`: coefficient `0.003116` (raises CT win probability)
- `lag_08__CT_place_MINI`: coefficient `0.003098` (raises CT win probability)
- `lag_00__T_place_SILO`: coefficient `-0.002791` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.002745` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21321`, seconds `96.50`, LSTM delta `-0.3141`

Top all feature movements:
- `lag_02__T_place_DECON`: contribution `-0.038626`
- `lag_03__T_place_MINI`: contribution `-0.035976`
- `lag_00__kill_diff_last_3s`: contribution `-0.013943`
- `lag_09__CT_place_HEAVEN`: contribution `-0.011378`
- `lag_02__CT_place_HEAVEN`: contribution `-0.010522`

Top utility-only movements:
- `lag_00__CT3__flash`: contribution `-0.004412`

### tick `21161`, seconds `94.00`, LSTM delta `+0.2755`

Top all feature movements:
- `lag_13__T_place_VENTS`: contribution `+0.042024`
- `lag_00__CT_shots_fired_sum`: contribution `+0.016679`
- `lag_00__kill_diff_last_3s`: contribution `+0.013943`
- `lag_00__CT_kills_last_3s`: contribution `+0.013576`
- `lag_13__CT_place_HEAVEN`: contribution `+0.010648`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21193`, seconds `94.50`, LSTM delta `-0.2267`

Top all feature movements:
- `lag_08__T_place_VENTS`: contribution `-0.019762`
- `lag_00__CT_shots_fired_sum`: contribution `-0.016679`
- `lag_00__kill_diff_last_3s`: contribution `-0.013943`
- `lag_00__CT_duck_amount_mean`: contribution `-0.009189`
- `lag_14__T_place_VENTS`: contribution `-0.009058`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21961`, seconds `106.50`, LSTM delta `+0.2240`

Top all feature movements:
- `lag_07__T_bomb_zone_count`: contribution `+0.022859`
- `lag_00__kill_diff_last_3s`: contribution `+0.013943`
- `lag_00__CT_kills_last_3s`: contribution `+0.013576`
- `lag_05__CT_duck_amount_mean`: contribution `+0.008805`
- `lag_14__T4__duck_amount`: contribution `+0.007758`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22281`, seconds `111.50`, LSTM delta `+0.1780`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.056725`
- `lag_00__CT_duck_amount_mean`: contribution `+0.027568`
- `lag_00__CT4__duck_amount`: contribution `+0.009351`
- `lag_00__CT_velocity_mean`: contribution `+0.009084`
- `lag_14__CT_duck_amount_mean`: contribution `+0.008699`

Top utility-only movements:
- No utility movement among the top local contributors.
