# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-g2-vs-betboom-bo3-pCfbtiY01aL_JW2Hy1pnZ6/g2-vs-betboom-m1-anubis.csv`
- round_num: `10`

## Largest probability jumps

- tick `90035`, seconds `43.00`, LSTM `0.4627`, delta `+0.3363`
- tick `88851`, seconds `24.50`, LSTM `0.1245`, delta `-0.1766`
- tick `90931`, seconds `57.00`, LSTM `0.1557`, delta `-0.1342`
- tick `88499`, seconds `19.00`, LSTM `0.3801`, delta `-0.1106`
- tick `88691`, seconds `22.00`, LSTM `0.3549`, delta `+0.0943`
- tick `90963`, seconds `57.50`, LSTM `0.0691`, delta `-0.0866`
- tick `89939`, seconds `41.50`, LSTM `0.1481`, delta `-0.0746`
- tick `90067`, seconds `43.50`, LSTM `0.5250`, delta `+0.0623`
- tick `88531`, seconds `19.50`, LSTM `0.3232`, delta `-0.0569`
- tick `89683`, seconds `37.50`, LSTM `0.2344`, delta `-0.0435`

## Top 15 local ridge features

- `lag_14__T_place_MAIN`: coefficient `-0.002983`, |coef| `0.002983`
- `lag_06__CT_place_TUNNELSTAIRS`: coefficient `-0.002520`, |coef| `0.002520`
- `lag_00__CT_shots_fired_sum`: coefficient `0.002301`, |coef| `0.002301`
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.002114`, |coef| `0.002114`
- `lag_14__CT_place_SNIPERSNEST`: coefficient `-0.001918`, |coef| `0.001918`
- `lag_03__CT_place_HEAVEN`: coefficient `-0.001913`, |coef| `0.001913`
- `lag_09__T_place_MAIN`: coefficient `-0.001910`, |coef| `0.001910`
- `lag_01__T_shots_fired_sum`: coefficient `-0.001883`, |coef| `0.001883`
- `lag_07__CT_place_TUNNEL`: coefficient `0.001882`, |coef| `0.001882`
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `0.001859`, |coef| `0.001859`
- `lag_01__CT_shots_fired_sum`: coefficient `0.001641`, |coef| `0.001641`
- `lag_00__kill_diff_last_3s`: coefficient `0.001605`, |coef| `0.001605`
- `lag_00__damage_diff_last_5s`: coefficient `0.001501`, |coef| `0.001501`
- `lag_00__T2__has_bomb`: coefficient `-0.001422`, |coef| `0.001422`
- `lag_08__CT3__duck_amount`: coefficient `0.001354`, |coef| `0.001354`

## Top 10 utility ridge features

- `lag_05__T_A_site_active_infernos`: coefficient `0.001213` (raises CT win probability)
- `lag_08__T1__molly`: coefficient `-0.001070` (lowers CT win probability)
- `lag_13__T_A_site_active_infernos`: coefficient `-0.001047` (lowers CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.001039` (raises CT win probability)
- `lag_11__T_B_site_active_smokes`: coefficient `-0.000959` (lowers CT win probability)
- `lag_05__T_active_infernos`: coefficient `0.000892` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.000876` (raises CT win probability)
- `lag_14__CT_A_site_active_infernos`: coefficient `-0.000832` (lowers CT win probability)
- `lag_04__T_A_site_active_smokes`: coefficient `-0.000773` (lowers CT win probability)
- `lag_07__T5__flash_duration`: coefficient `-0.000728` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__T_place_MAIN`: coefficient `-0.002983` (lowers CT win probability)
- `lag_06__CT_place_TUNNELSTAIRS`: coefficient `-0.002520` (lowers CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.002301` (raises CT win probability)
- `lag_00__T_place_FOUNTAIN`: coefficient `-0.002114` (lowers CT win probability)
- `lag_14__CT_place_SNIPERSNEST`: coefficient `-0.001918` (lowers CT win probability)
- `lag_03__CT_place_HEAVEN`: coefficient `-0.001913` (lowers CT win probability)
- `lag_09__T_place_MAIN`: coefficient `-0.001910` (lowers CT win probability)
- `lag_01__T_shots_fired_sum`: coefficient `-0.001883` (lowers CT win probability)
- `lag_07__CT_place_TUNNEL`: coefficient `0.001882` (raises CT win probability)
- `lag_11__CT_place_TUNNELSTAIRS`: coefficient `0.001859` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `90035`, seconds `43.00`, LSTM delta `+0.3363`

Top all feature movements:
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `+0.035489`
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `+0.026183`
- `lag_14__T_place_MAIN`: contribution `+0.019288`
- `lag_00__CT_shots_fired_sum`: contribution `+0.015987`
- `lag_03__CT_place_HEAVEN`: contribution `+0.010331`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `88851`, seconds `24.50`, LSTM delta `-0.1766`

Top all feature movements:
- `lag_03__CT_place_BRICKS`: contribution `-0.021668`
- `lag_07__CT_place_BRICKS`: contribution `-0.021421`
- `lag_09__T_place_MAIN`: contribution `-0.012348`
- `lag_05__CT_place_BRICKS`: contribution `-0.011041`
- `lag_00__CT_place_BRICKS`: contribution `+0.009365`

Top utility-only movements:
- `lag_07__T5__flash_duration`: contribution `-0.005979`
- `lag_07__T4__flash_duration`: contribution `-0.004197`
- `lag_07__T_flash_duration_sum`: contribution `-0.003577`

### tick `90931`, seconds `57.00`, LSTM delta `-0.1342`

Top all feature movements:
- `lag_06__CT_place_TUNNELSTAIRS`: contribution `-0.035489`
- `lag_12__T_place_WALKWAY`: contribution `-0.012661`
- `lag_05__T_place_WALKWAY`: contribution `-0.012414`
- `lag_15__CT_place_TUNNEL`: contribution `-0.009699`
- `lag_01__T_place_WALKWAY`: contribution `+0.009446`

Top utility-only movements:
- `lag_02__T1__flash_duration`: contribution `-0.003201`

### tick `88499`, seconds `19.00`, LSTM delta `-0.1106`

Top all feature movements:
- `lag_01__CT_place_FOUNTAIN`: contribution `-0.012288`
- `lag_10__CT_place_MAIN`: contribution `-0.006305`
- `lag_01__T_shots_fired_sum`: contribution `-0.005647`
- `lag_00__CT_shots_fired_sum`: contribution `-0.004796`
- `lag_00__kill_diff_last_3s`: contribution `-0.003863`

Top utility-only movements:
- `lag_14__CT_A_site_active_infernos`: contribution `-0.002937`
- `lag_15__CT5__molly`: contribution `-0.001752`

### tick `88691`, seconds `22.00`, LSTM delta `+0.0943`

Top all feature movements:
- `lag_05__CT_place_FOUNTAIN`: contribution `+0.012990`
- `lag_02__CT_place_BRICKS`: contribution `+0.012410`
- `lag_00__CT_place_BRICKS`: contribution `+0.009365`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007993`
- `lag_07__CT_place_FOUNTAIN`: contribution `+0.006304`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `+0.005189`
- `lag_02__T_flash_duration_sum`: contribution `+0.003950`
