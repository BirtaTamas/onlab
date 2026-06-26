# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `1`

## Largest probability jumps

- tick `14884`, seconds `59.00`, LSTM `0.6028`, delta `+0.3823`
- tick `15140`, seconds `63.00`, LSTM `0.4327`, delta `-0.3782`
- tick `14820`, seconds `58.00`, LSTM `0.4741`, delta `+0.2997`
- tick `14852`, seconds `58.50`, LSTM `0.2205`, delta `-0.2536`
- tick `15108`, seconds `62.50`, LSTM `0.8109`, delta `+0.2455`
- tick `14628`, seconds `55.00`, LSTM `0.0997`, delta `-0.1914`
- tick `15044`, seconds `61.50`, LSTM `0.6188`, delta `-0.1376`
- tick `15204`, seconds `64.00`, LSTM `0.3970`, delta `-0.1209`
- tick `14724`, seconds `56.50`, LSTM `0.1678`, delta `+0.1126`
- tick `14916`, seconds `59.50`, LSTM `0.6988`, delta `+0.0960`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003458`, |coef| `0.003458`
- `lag_00__damage_diff_last_5s`: coefficient `0.003457`, |coef| `0.003457`
- `lag_07__T_place_UPPERPARK`: coefficient `-0.002792`, |coef| `0.002792`
- `lag_00__CT_damage_last_5s`: coefficient `0.002761`, |coef| `0.002761`
- `lag_05__T_place_UPPERPARK`: coefficient `-0.002716`, |coef| `0.002716`
- `lag_00__CT_kills_last_3s`: coefficient `0.002435`, |coef| `0.002435`
- `lag_07__CT_place_BACKOFA`: coefficient `-0.002058`, |coef| `0.002058`
- `lag_02__kill_diff_last_3s`: coefficient `0.001923`, |coef| `0.001923`
- `lag_11__CT1__flash_duration`: coefficient `0.001903`, |coef| `0.001903`
- `lag_00__CT_place_LOWERPARK`: coefficient `0.001888`, |coef| `0.001888`
- `lag_00__T_kills_last_3s`: coefficient `-0.001879`, |coef| `0.001879`
- `lag_06__CT_place_STAIRS`: coefficient `-0.001871`, |coef| `0.001871`
- `lag_03__CT_kills_last_3s`: coefficient `0.001865`, |coef| `0.001865`
- `lag_14__CT_place_LOBBY`: coefficient `-0.001830`, |coef| `0.001830`
- `lag_14__CT5__flash_duration`: coefficient `-0.001786`, |coef| `0.001786`

## Top 10 utility ridge features

- `lag_11__CT1__flash_duration`: coefficient `0.001903` (raises CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `-0.001786` (lowers CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `0.001651` (raises CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `-0.001567` (lowers CT win probability)
- `lag_03__CT1__flash_duration`: coefficient `-0.001540` (lowers CT win probability)
- `lag_12__T5__flash_duration`: coefficient `-0.001479` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.001462` (lowers CT win probability)
- `lag_09__T1__flash_duration`: coefficient `0.001387` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001325` (raises CT win probability)
- `lag_11__T5__flash_duration`: coefficient `0.001290` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003458` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003457` (raises CT win probability)
- `lag_07__T_place_UPPERPARK`: coefficient `-0.002792` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002761` (raises CT win probability)
- `lag_05__T_place_UPPERPARK`: coefficient `-0.002716` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002435` (raises CT win probability)
- `lag_07__CT_place_BACKOFA`: coefficient `-0.002058` (lowers CT win probability)
- `lag_02__kill_diff_last_3s`: coefficient `0.001923` (raises CT win probability)
- `lag_00__CT_place_LOWERPARK`: coefficient `0.001888` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001879` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `14884`, seconds `59.00`, LSTM delta `+0.3823`

Top all feature movements:
- `lag_07__T_place_UPPERPARK`: contribution `+0.014724`
- `lag_05__T_place_UPPERPARK`: contribution `+0.014321`
- `lag_15__CT_place_LOBBY`: contribution `+0.013626`
- `lag_01__CT_place_STAIRS`: contribution `+0.012905`
- `lag_10__T1__flash_duration`: contribution `+0.010166`

Top utility-only movements:
- `lag_10__T1__flash_duration`: contribution `+0.010166`
- `lag_14__CT5__flash_duration`: contribution `+0.009697`
- `lag_12__T5__flash_duration`: contribution `+0.009453`
- `lag_03__CT1__flash_duration`: contribution `+0.006874`

### tick `15140`, seconds `63.00`, LSTM delta `-0.3782`

Top all feature movements:
- `lag_07__CT_place_BACKOFA`: contribution `-0.019871`
- `lag_02__CT_place_BACKOFA`: contribution `-0.016029`
- `lag_07__CT_place_STAIRS`: contribution `-0.012019`
- `lag_09__CT_place_STAIRS`: contribution `-0.008620`
- `lag_11__CT1__flash_duration`: contribution `-0.008493`

Top utility-only movements:
- `lag_11__CT1__flash_duration`: contribution `-0.008493`

### tick `14820`, seconds `58.00`, LSTM delta `+0.2997`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.016645`
- `lag_07__T_place_UPPERPARK`: contribution `+0.014724`
- `lag_05__T_place_UPPERPARK`: contribution `+0.014321`
- `lag_00__damage_diff_last_5s`: contribution `+0.009280`
- `lag_13__CT_place_LOBBY`: contribution `+0.008729`

Top utility-only movements:
- `lag_08__CT1__flash_duration`: contribution `+0.007370`
- `lag_08__T1__flash_duration`: contribution `+0.007173`
- `lag_01__CT1__flash_duration`: contribution `+0.005449`
- `lag_12__CT5__flash_duration`: contribution `+0.005176`
- `lag_14__T3__flash_duration`: contribution `+0.004526`

### tick `14852`, seconds `58.50`, LSTM delta `-0.2536`

Top all feature movements:
- `lag_14__CT_place_LOBBY`: contribution `-0.014977`
- `lag_00__CT_place_STAIRS`: contribution `-0.011410`
- `lag_09__T1__flash_duration`: contribution `-0.009649`
- `lag_00__CT_place_LOWERPARK`: contribution `-0.008437`
- `lag_01__kill_diff_last_3s`: contribution `-0.008348`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `-0.009649`
- `lag_11__T5__flash_duration`: contribution `-0.008242`

### tick `15108`, seconds `62.50`, LSTM delta `+0.2455`

Top all feature movements:
- `lag_06__CT_place_STAIRS`: contribution `+0.014559`
- `lag_01__CT_place_BACKOFA`: contribution `+0.010815`
- `lag_08__CT_place_STAIRS`: contribution `+0.009579`
- `lag_00__kill_diff_last_3s`: contribution `+0.008323`
- `lag_00__T_place_UPPERPARK`: contribution `+0.007293`

Top utility-only movements:
- No utility movement among the top local contributors.
