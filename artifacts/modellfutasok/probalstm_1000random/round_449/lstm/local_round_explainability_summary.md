# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `94090`, seconds `25.50`, LSTM `0.2279`, delta `-0.2925`
- tick `93930`, seconds `23.00`, LSTM `0.5305`, delta `+0.1614`
- tick `94058`, seconds `25.00`, LSTM `0.5203`, delta `+0.0973`
- tick `94602`, seconds `33.50`, LSTM `0.2157`, delta `+0.0787`
- tick `94122`, seconds `26.00`, LSTM `0.1502`, delta `-0.0777`
- tick `95306`, seconds `44.50`, LSTM `0.1287`, delta `-0.0737`
- tick `93418`, seconds `15.00`, LSTM `0.3872`, delta `-0.0555`
- tick `93962`, seconds `23.50`, LSTM `0.4765`, delta `-0.0540`
- tick `95882`, seconds `53.50`, LSTM `0.0150`, delta `-0.0414`
- tick `92810`, seconds `5.50`, LSTM `0.3894`, delta `-0.0394`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001727`, |coef| `0.001727`
- `lag_13__T_flashes_last_5s`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_03__T_flashes_last_5s`: coefficient `0.001490`, |coef| `0.001490`
- `lag_02__CT5__flash_duration`: coefficient `-0.001318`, |coef| `0.001318`
- `lag_08__CT_place_UPPERTUNNEL`: coefficient `-0.001256`, |coef| `0.001256`
- `lag_00__kill_diff_last_3s`: coefficient `0.001231`, |coef| `0.001231`
- `lag_01__CT5__flash_duration`: coefficient `-0.001204`, |coef| `0.001204`
- `lag_04__CT_shots_fired_sum`: coefficient `0.001173`, |coef| `0.001173`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001076`, |coef| `0.001076`
- `lag_11__CT2__is_walking`: coefficient `-0.000959`, |coef| `0.000959`
- `lag_04__CT_place_PIT`: coefficient `0.000934`, |coef| `0.000934`
- `lag_11__CT3__flash_duration`: coefficient `-0.000930`, |coef| `0.000930`
- `lag_00__T_kills_last_3s`: coefficient `-0.000896`, |coef| `0.000896`
- `lag_00__CT1__flash_duration`: coefficient `-0.000893`, |coef| `0.000893`
- `lag_01__CT3__shots_fired`: coefficient `-0.000890`, |coef| `0.000890`

## Top 10 utility ridge features

- `lag_13__T_flashes_last_5s`: coefficient `-0.001637` (lowers CT win probability)
- `lag_03__T_flashes_last_5s`: coefficient `0.001490` (raises CT win probability)
- `lag_02__CT5__flash_duration`: coefficient `-0.001318` (lowers CT win probability)
- `lag_01__CT5__flash_duration`: coefficient `-0.001204` (lowers CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.000930` (lowers CT win probability)
- `lag_00__CT1__flash_duration`: coefficient `-0.000893` (lowers CT win probability)
- `lag_14__T_flash_duration_sum`: coefficient `-0.000848` (lowers CT win probability)
- `lag_09__T5__flash_duration`: coefficient `0.000842` (raises CT win probability)
- `lag_14__T1__flash_duration`: coefficient `-0.000832` (lowers CT win probability)
- `lag_04__T1__flash_duration`: coefficient `0.000807` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001727` (raises CT win probability)
- `lag_08__CT_place_UPPERTUNNEL`: coefficient `-0.001256` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001231` (raises CT win probability)
- `lag_04__CT_shots_fired_sum`: coefficient `0.001173` (raises CT win probability)
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.001076` (raises CT win probability)
- `lag_11__CT2__is_walking`: coefficient `-0.000959` (lowers CT win probability)
- `lag_04__CT_place_PIT`: coefficient `0.000934` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000896` (lowers CT win probability)
- `lag_01__CT3__shots_fired`: coefficient `-0.000890` (lowers CT win probability)
- `lag_09__CT5__is_scoped`: coefficient `-0.000823` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `94090`, seconds `25.50`, LSTM delta `-0.2925`

Top all feature movements:
- `lag_13__T_flashes_last_5s`: contribution `-0.014831`
- `lag_03__T_flashes_last_5s`: contribution `-0.013503`
- `lag_04__CT_shots_fired_sum`: contribution `-0.009781`
- `lag_08__CT_place_UPPERTUNNEL`: contribution `-0.009635`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007199`

Top utility-only movements:
- `lag_13__T_flashes_last_5s`: contribution `-0.014831`
- `lag_03__T_flashes_last_5s`: contribution `-0.013503`
- `lag_14__T_flash_duration_sum`: contribution `-0.005701`
- `lag_14__T1__flash_duration`: contribution `-0.004545`
- `lag_04__T1__flash_duration`: contribution `-0.004412`

### tick `93930`, seconds `23.00`, LSTM delta `+0.1614`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.010799`
- `lag_08__T_flashes_last_5s`: contribution `+0.006588`
- `lag_03__CT_place_UPPERTUNNEL`: contribution `+0.004806`
- `lag_00__CT1__flash_duration`: contribution `+0.004291`
- `lag_09__T5__flash_duration`: contribution `+0.004283`

Top utility-only movements:
- `lag_08__T_flashes_last_5s`: contribution `+0.006588`
- `lag_00__CT1__flash_duration`: contribution `+0.004291`
- `lag_09__T5__flash_duration`: contribution `+0.004283`
- `lag_12__CT3__flash_duration`: contribution `+0.003897`
- `lag_06__CT3__flash_duration`: contribution `+0.003376`

### tick `94058`, seconds `25.00`, LSTM delta `+0.0973`

Top all feature movements:
- `lag_04__CT_shots_fired_sum`: contribution `+0.007336`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007199`
- `lag_02__CT5__flash_duration`: contribution `+0.005349`
- `lag_13__T_flashed_players`: contribution `+0.003960`
- `lag_11__CT_flashed_players`: contribution `+0.003226`

Top utility-only movements:
- `lag_02__CT5__flash_duration`: contribution `+0.005349`
- `lag_13__T_flash_duration_sum`: contribution `+0.003080`
- `lag_02__CT_flash_duration_sum`: contribution `+0.001681`
- `lag_13__T1__flash_duration`: contribution `+0.001624`
- `lag_02__T_flashes_last_5s`: contribution `+0.001496`

### tick `94602`, seconds `33.50`, LSTM delta `+0.0787`

Top all feature movements:
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.015150`
- `lag_00__CT_place_UPPERTUNNEL`: contribution `+0.005325`
- `lag_00__T1__duck_amount`: contribution `+0.002085`
- `lag_08__T1__duck_amount`: contribution `+0.002034`
- `lag_04__T3__duck_amount`: contribution `+0.001694`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `+0.000870`

### tick `94122`, seconds `26.00`, LSTM delta `-0.0777`

Top all feature movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.005995`
- `lag_04__T_flashes_last_5s`: contribution `-0.005535`
- `lag_06__CT_shots_fired_sum`: contribution `-0.003789`
- `lag_09__CT_place_UPPERTUNNEL`: contribution `-0.003558`
- `lag_14__CT_flashed_players`: contribution `-0.003473`

Top utility-only movements:
- `lag_14__T_flashes_last_5s`: contribution `-0.005995`
- `lag_04__T_flashes_last_5s`: contribution `-0.005535`
- `lag_12__CT3__flash_duration`: contribution `-0.003409`
- `lag_04__CT3__flash_duration`: contribution `-0.002286`
- `lag_04__CT5__flash_duration`: contribution `+0.002138`
