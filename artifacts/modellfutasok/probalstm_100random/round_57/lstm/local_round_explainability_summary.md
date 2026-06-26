# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `61533`, seconds `84.50`, LSTM `0.8828`, delta `+0.2768`
- tick `62141`, seconds `94.00`, LSTM `0.5210`, delta `-0.2754`
- tick `62269`, seconds `96.00`, LSTM `0.8241`, delta `+0.2591`
- tick `62461`, seconds `99.00`, LSTM `0.5702`, delta `-0.2577`
- tick `61277`, seconds `80.50`, LSTM `0.7677`, delta `+0.2470`
- tick `58173`, seconds `32.00`, LSTM `0.6371`, delta `+0.2128`
- tick `61437`, seconds `83.00`, LSTM `0.6208`, delta `-0.1334`
- tick `61725`, seconds `87.50`, LSTM `0.8017`, delta `-0.1140`
- tick `62173`, seconds `94.50`, LSTM `0.5917`, delta `+0.0706`
- tick `60989`, seconds `76.00`, LSTM `0.5753`, delta `+0.0610`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005211`, |coef| `0.005211`
- `lag_00__CT_kills_last_3s`: coefficient `0.004653`, |coef| `0.004653`
- `lag_12__T5__flash_duration`: coefficient `0.003046`, |coef| `0.003046`
- `lag_09__CT_shots_fired_sum`: coefficient `0.002837`, |coef| `0.002837`
- `lag_13__CT1__shots_fired`: coefficient `-0.002711`, |coef| `0.002711`
- `lag_00__damage_diff_last_5s`: coefficient `0.002688`, |coef| `0.002688`
- `lag_13__CT_shots_fired_sum`: coefficient `-0.002636`, |coef| `0.002636`
- `lag_05__CT2__flash_duration`: coefficient `0.002332`, |coef| `0.002332`
- `lag_09__CT1__shots_fired`: coefficient `0.002169`, |coef| `0.002169`
- `lag_04__T_shots_fired_sum`: coefficient `-0.002132`, |coef| `0.002132`
- `lag_12__CT_place_UNDERA`: coefficient `0.002112`, |coef| `0.002112`
- `lag_09__CT_place_EXTENDEDA`: coefficient `0.002035`, |coef| `0.002035`
- `lag_09__CT_place_HOLE`: coefficient `0.001953`, |coef| `0.001953`
- `lag_06__CT4__duck_amount`: coefficient `-0.001859`, |coef| `0.001859`
- `lag_00__CT_damage_last_5s`: coefficient `0.001829`, |coef| `0.001829`

## Top 10 utility ridge features

- `lag_12__T5__flash_duration`: coefficient `0.003046` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `0.002332` (raises CT win probability)
- `lag_12__T_flash_duration_sum`: coefficient `0.001780` (raises CT win probability)
- `lag_12__T4__flash_duration`: coefficient `0.001326` (raises CT win probability)
- `lag_07__CT5__flash_duration`: coefficient `0.001256` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `0.001245` (raises CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `0.001228` (raises CT win probability)
- `lag_14__T5__flash_duration`: coefficient `0.001199` (raises CT win probability)
- `lag_06__T_B_site_active_infernos`: coefficient `0.001122` (raises CT win probability)
- `lag_06__T5__flash_duration`: coefficient `-0.001121` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005211` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004653` (raises CT win probability)
- `lag_09__CT_shots_fired_sum`: coefficient `0.002837` (raises CT win probability)
- `lag_13__CT1__shots_fired`: coefficient `-0.002711` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002688` (raises CT win probability)
- `lag_13__CT_shots_fired_sum`: coefficient `-0.002636` (lowers CT win probability)
- `lag_09__CT1__shots_fired`: coefficient `0.002169` (raises CT win probability)
- `lag_04__T_shots_fired_sum`: coefficient `-0.002132` (lowers CT win probability)
- `lag_12__CT_place_UNDERA`: coefficient `0.002112` (raises CT win probability)
- `lag_09__CT_place_EXTENDEDA`: coefficient `0.002035` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `61533`, seconds `84.50`, LSTM delta `+0.2768`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013435`
- `lag_00__kill_diff_last_3s`: contribution `+0.012542`
- `lag_07__CT_flashed_players`: contribution `+0.010615`
- `lag_04__T_shots_fired_sum`: contribution `+0.007993`
- `lag_06__CT4__duck_amount`: contribution `+0.006827`

Top utility-only movements:
- `lag_07__CT5__flash_duration`: contribution `+0.005825`
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.005544`
- `lag_07__CT_flash_duration_sum`: contribution `+0.003926`

### tick `62141`, seconds `94.00`, LSTM delta `-0.2754`

Top all feature movements:
- `lag_09__CT_shots_fired_sum`: contribution `-0.057157`
- `lag_09__CT1__shots_fired`: contribution `-0.032096`
- `lag_00__kill_diff_last_3s`: contribution `-0.012542`
- `lag_02__T5__flash_duration`: contribution `-0.010155`
- `lag_01__CT_place_BDOORS`: contribution `-0.008788`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `-0.010155`
- `lag_12__T4__flash_duration`: contribution `+0.007878`
- `lag_02__T4__flash_duration`: contribution `-0.005558`
- `lag_02__T_flash_duration_sum`: contribution `-0.004999`
- `lag_12__T_flash_duration_sum`: contribution `+0.004300`

### tick `62269`, seconds `96.00`, LSTM delta `+0.2591`

Top all feature movements:
- `lag_13__CT_shots_fired_sum`: contribution `+0.053118`
- `lag_13__CT1__shots_fired`: contribution `+0.040112`
- `lag_03__CT_place_HOLE`: contribution `+0.013910`
- `lag_00__CT_kills_last_3s`: contribution `+0.013435`
- `lag_00__kill_diff_last_3s`: contribution `+0.012542`

Top utility-only movements:
- `lag_06__T5__flash_duration`: contribution `+0.009145`
- `lag_06__T4__flash_duration`: contribution `+0.005107`
- `lag_06__T_flash_duration_sum`: contribution `+0.004691`

### tick `62461`, seconds `99.00`, LSTM delta `-0.2577`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.025085`
- `lag_12__T5__flash_duration`: contribution `-0.024837`
- `lag_09__CT_place_HOLE`: contribution `-0.021805`
- `lag_00__CT_kills_last_3s`: contribution `-0.013435`
- `lag_12__T_flash_duration_sum`: contribution `-0.010198`

Top utility-only movements:
- `lag_12__T5__flash_duration`: contribution `-0.024837`
- `lag_12__T_flash_duration_sum`: contribution `-0.010198`
- `lag_12__T4__flash_duration`: contribution `-0.007878`

### tick `61277`, seconds `80.50`, LSTM delta `+0.2470`

Top all feature movements:
- `lag_04__T_shots_fired_sum`: contribution `+0.014387`
- `lag_00__CT_kills_last_3s`: contribution `+0.013435`
- `lag_00__kill_diff_last_3s`: contribution `+0.012542`
- `lag_13__CT_place_ARAMP`: contribution `+0.008341`
- `lag_06__CT4__duck_amount`: contribution `+0.006626`

Top utility-only movements:
- No utility movement among the top local contributors.
