# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `6`

## Largest probability jumps

- tick `44578`, seconds `84.00`, LSTM `0.1073`, delta `-0.3101`
- tick `44514`, seconds `83.00`, LSTM `0.5028`, delta `+0.2579`
- tick `44450`, seconds `82.00`, LSTM `0.2505`, delta `-0.2322`
- tick `43394`, seconds `65.50`, LSTM `0.3279`, delta `-0.1574`
- tick `43362`, seconds `65.00`, LSTM `0.4854`, delta `-0.1488`
- tick `44354`, seconds `80.50`, LSTM `0.3741`, delta `+0.1317`
- tick `44546`, seconds `83.50`, LSTM `0.4175`, delta `-0.0853`
- tick `44386`, seconds `81.00`, LSTM `0.4543`, delta `+0.0803`
- tick `43682`, seconds `70.00`, LSTM `0.3076`, delta `+0.0679`
- tick `44066`, seconds `76.00`, LSTM `0.2776`, delta `-0.0666`

## Top 15 local ridge features

- `lag_10__CT_place_BACKALLEY`: coefficient `-0.003224`, |coef| `0.003224`
- `lag_00__kill_diff_last_3s`: coefficient `0.003093`, |coef| `0.003093`
- `lag_01__T_shots_fired_sum`: coefficient `-0.002644`, |coef| `0.002644`
- `lag_09__CT_place_BACKALLEY`: coefficient `-0.002214`, |coef| `0.002214`
- `lag_01__T4__shots_fired`: coefficient `-0.002155`, |coef| `0.002155`
- `lag_00__T_kills_last_3s`: coefficient `-0.002040`, |coef| `0.002040`
- `lag_00__T_place_TRUCK`: coefficient `-0.001966`, |coef| `0.001966`
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001898`, |coef| `0.001898`
- `lag_00__CT_kills_last_3s`: coefficient `0.001852`, |coef| `0.001852`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001713`, |coef| `0.001713`
- `lag_10__CT2__duck_amount`: coefficient `-0.001708`, |coef| `0.001708`
- `lag_01__CT_place_BACKALLEY`: coefficient `0.001663`, |coef| `0.001663`
- `lag_01__CT_place_SHOP`: coefficient `-0.001641`, |coef| `0.001641`
- `lag_14__T_flashed_players`: coefficient `0.001639`, |coef| `0.001639`
- `lag_06__CT2__duck_amount`: coefficient `-0.001619`, |coef| `0.001619`

## Top 10 utility ridge features

- `lag_04__CT2__flash_duration`: coefficient `-0.001328` (lowers CT win probability)
- `lag_06__T4__flash_duration`: coefficient `0.001139` (raises CT win probability)
- `lag_00__CT2__flash_duration`: coefficient `-0.001135` (lowers CT win probability)
- `lag_11__CT2__flash_duration`: coefficient `0.001076` (raises CT win probability)
- `lag_02__T2__flash`: coefficient `0.000934` (raises CT win probability)
- `lag_03__CT2__flash_duration`: coefficient `-0.000831` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.000791` (raises CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000760` (raises CT win probability)
- `lag_12__CT2__flash_duration`: coefficient `0.000699` (raises CT win probability)
- `lag_06__CT2__flash_duration`: coefficient `0.000689` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_10__CT_place_BACKALLEY`: coefficient `-0.003224` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003093` (raises CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.002644` (lowers CT win probability)
- `lag_09__CT_place_BACKALLEY`: coefficient `-0.002214` (lowers CT win probability)
- `lag_01__T4__shots_fired`: coefficient `-0.002155` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002040` (lowers CT win probability)
- `lag_00__T_place_TRUCK`: coefficient `-0.001966` (lowers CT win probability)
- `lag_00__CT_place_BACKALLEY`: coefficient `0.001898` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001852` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001713` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `44578`, seconds `84.00`, LSTM delta `-0.3101`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.014891`
- `lag_01__T_shots_fired_sum`: contribution `-0.013874`
- `lag_01__T4__shots_fired`: contribution `-0.009319`
- `lag_00__T_shots_fired_sum`: contribution `-0.007707`
- `lag_00__T_kills_last_3s`: contribution `-0.006461`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44514`, seconds `83.00`, LSTM delta `+0.2579`

Top all feature movements:
- `lag_01__T_shots_fired_sum`: contribution `+0.015857`
- `lag_01__T4__shots_fired`: contribution `+0.010650`
- `lag_01__CT_place_SHOP`: contribution `+0.008230`
- `lag_00__kill_diff_last_3s`: contribution `+0.007446`
- `lag_06__CT2__duck_amount`: contribution `+0.006168`

Top utility-only movements:
- `lag_04__CT2__flash_duration`: contribution `+0.005207`

### tick `44450`, seconds `82.00`, LSTM delta `-0.2322`

Top all feature movements:
- `lag_14__T_flashed_players`: contribution `-0.009486`
- `lag_00__T_shots_fired_sum`: contribution `-0.008991`
- `lag_00__kill_diff_last_3s`: contribution `-0.007446`
- `lag_10__CT2__duck_amount`: contribution `-0.006508`
- `lag_00__T_kills_last_3s`: contribution `-0.006461`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `43394`, seconds `65.50`, LSTM delta `-0.1574`

Top all feature movements:
- `lag_10__CT_place_BACKALLEY`: contribution `-0.048328`
- `lag_01__CT_place_BACKALLEY`: contribution `-0.024930`
- `lag_02__T_place_CATWALK`: contribution `-0.008189`
- `lag_03__CT_place_JUNGLE`: contribution `-0.006103`
- `lag_03__CT_place_SHOP`: contribution `-0.004310`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `43362`, seconds `65.00`, LSTM delta `-0.1488`

Top all feature movements:
- `lag_09__CT_place_BACKALLEY`: contribution `-0.033195`
- `lag_00__CT_place_BACKALLEY`: contribution `-0.028453`
- `lag_00__kill_diff_last_3s`: contribution `-0.007446`
- `lag_00__T_kills_last_3s`: contribution `-0.006461`
- `lag_01__T_place_CATWALK`: contribution `-0.005111`

Top utility-only movements:
- No utility movement among the top local contributors.
