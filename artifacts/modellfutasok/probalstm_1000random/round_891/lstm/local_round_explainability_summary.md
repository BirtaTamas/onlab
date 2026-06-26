# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m1-anubis.csv`
- round_num: `14`

## Largest probability jumps

- tick `118707`, seconds `70.00`, LSTM `0.8290`, delta `+0.2247`
- tick `118323`, seconds `64.00`, LSTM `0.8222`, delta `+0.1737`
- tick `117491`, seconds `51.00`, LSTM `0.4267`, delta `+0.1693`
- tick `118995`, seconds `74.50`, LSTM `0.5751`, delta `-0.1544`
- tick `118547`, seconds `67.50`, LSTM `0.5656`, delta `-0.1455`
- tick `118355`, seconds `64.50`, LSTM `0.6870`, delta `-0.1352`
- tick `119219`, seconds `78.00`, LSTM `0.7521`, delta `+0.1196`
- tick `118739`, seconds `70.50`, LSTM `0.7210`, delta `-0.1079`
- tick `114291`, seconds `1.00`, LSTM `0.2897`, delta `+0.0865`
- tick `118259`, seconds `63.00`, LSTM `0.5651`, delta `+0.0850`

## Top 15 local ridge features

- `lag_02__CT_place_BRIDGE`: coefficient `0.002379`, |coef| `0.002379`
- `lag_00__kill_diff_last_3s`: coefficient `0.002142`, |coef| `0.002142`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002118`, |coef| `0.002118`
- `lag_02__CT_place_CTSIDEUPPER`: coefficient `0.002065`, |coef| `0.002065`
- `lag_02__T_shots_fired_sum`: coefficient `0.001931`, |coef| `0.001931`
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `0.001918`, |coef| `0.001918`
- `lag_00__T5__shots_fired`: coefficient `0.001902`, |coef| `0.001902`
- `lag_01__CT3__shots_fired`: coefficient `0.001822`, |coef| `0.001822`
- `lag_02__T5__shots_fired`: coefficient `0.001810`, |coef| `0.001810`
- `lag_00__CT_kills_last_3s`: coefficient `0.001803`, |coef| `0.001803`
- `lag_07__T4__shots_fired`: coefficient `-0.001757`, |coef| `0.001757`
- `lag_08__T4__shots_fired`: coefficient `0.001673`, |coef| `0.001673`
- `lag_03__T5__flash_duration`: coefficient `0.001671`, |coef| `0.001671`
- `lag_04__CT_place_PALACEINTERIOR`: coefficient `-0.001651`, |coef| `0.001651`
- `lag_07__T_shots_fired_sum`: coefficient `-0.001646`, |coef| `0.001646`

## Top 10 utility ridge features

- `lag_03__T5__flash_duration`: coefficient `0.001671` (raises CT win probability)
- `lag_00__CT_smokes_last_5s`: coefficient `0.000978` (raises CT win probability)
- `lag_06__T_A_site_active_infernos`: coefficient `0.000766` (raises CT win probability)
- `lag_01__CT_smokes_last_5s`: coefficient `0.000726` (raises CT win probability)
- `lag_03__T_flash_duration_sum`: coefficient `0.000666` (raises CT win probability)
- `lag_10__T2__molly`: coefficient `-0.000645` (lowers CT win probability)
- `lag_14__T5__molly`: coefficient `-0.000589` (lowers CT win probability)
- `lag_13__T2__molly`: coefficient `-0.000581` (lowers CT win probability)
- `lag_11__T_active_infernos`: coefficient `0.000576` (raises CT win probability)
- `lag_12__CT_smokes_last_5s`: coefficient `-0.000565` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_BRIDGE`: coefficient `0.002379` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002142` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.002118` (raises CT win probability)
- `lag_02__CT_place_CTSIDEUPPER`: coefficient `0.002065` (raises CT win probability)
- `lag_02__T_shots_fired_sum`: coefficient `0.001931` (raises CT win probability)
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `0.001918` (raises CT win probability)
- `lag_00__T5__shots_fired`: coefficient `0.001902` (raises CT win probability)
- `lag_01__CT3__shots_fired`: coefficient `0.001822` (raises CT win probability)
- `lag_02__T5__shots_fired`: coefficient `0.001810` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001803` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `118707`, seconds `70.00`, LSTM delta `+0.2247`

Top all feature movements:
- `lag_07__T_shots_fired_sum`: contribution `+0.034548`
- `lag_07__T4__shots_fired`: contribution `+0.030386`
- `lag_12__CT_place_FOUNTAIN`: contribution `+0.012864`
- `lag_12__CT_shots_fired_sum`: contribution `+0.008043`
- `lag_01__CT_shots_fired_sum`: contribution `+0.007358`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118323`, seconds `64.00`, LSTM delta `+0.1737`

Top all feature movements:
- `lag_00__CT_place_FOUNTAIN`: contribution `+0.016072`
- `lag_00__CT_shots_fired_sum`: contribution `-0.010621`
- `lag_01__CT_shots_fired_sum`: contribution `+0.010301`
- `lag_04__CT_place_PALACEINTERIOR`: contribution `+0.006729`
- `lag_01__CT3__shots_fired`: contribution `+0.006559`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `117491`, seconds `51.00`, LSTM delta `+0.1693`

Top all feature movements:
- `lag_02__CT_place_BRIDGE`: contribution `+0.027264`
- `lag_03__T5__flash_duration`: contribution `+0.012217`
- `lag_02__T_shots_fired_sum`: contribution `+0.008688`
- `lag_00__T5__shots_fired`: contribution `+0.007015`
- `lag_02__T5__shots_fired`: contribution `+0.006675`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.012217`
- `lag_03__T_flash_duration_sum`: contribution `+0.001977`

### tick `118995`, seconds `74.50`, LSTM delta `-0.1544`

Top all feature movements:
- `lag_15__CT_place_LOWERTUNNEL`: contribution `-0.014099`
- `lag_08__CT_shots_fired_sum`: contribution `-0.011563`
- `lag_08__CT2__shots_fired`: contribution `-0.008491`
- `lag_01__T_bomb_zone_count`: contribution `-0.007829`
- `lag_04__CT_place_PALACEINTERIOR`: contribution `-0.006729`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118547`, seconds `67.50`, LSTM delta `-0.1455`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.040545`
- `lag_02__T4__shots_fired`: contribution `-0.021814`
- `lag_07__T_shots_fired_sum`: contribution `+0.008637`
- `lag_03__CT_place_LOWERTUNNEL`: contribution `-0.007591`
- `lag_08__CT_shots_fired_sum`: contribution `+0.006745`

Top utility-only movements:
- No utility movement among the top local contributors.
