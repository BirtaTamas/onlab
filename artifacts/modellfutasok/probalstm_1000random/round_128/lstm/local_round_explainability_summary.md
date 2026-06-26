# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-b8-bo3-rUWlZLFFckLiQv1C1wSlHb/g2-vs-b8-m3-ancient.csv`
- round_num: `7`

## Largest probability jumps

- tick `62362`, seconds `105.50`, LSTM `0.1297`, delta `-0.3798`
- tick `62234`, seconds `103.50`, LSTM `0.5182`, delta `+0.3003`
- tick `62074`, seconds `101.00`, LSTM `0.3951`, delta `-0.1509`
- tick `62106`, seconds `101.50`, LSTM `0.2787`, delta `-0.1163`
- tick `62138`, seconds `102.00`, LSTM `0.2110`, delta `-0.0677`
- tick `62586`, seconds `109.00`, LSTM `0.0185`, delta `-0.0596`
- tick `61658`, seconds `94.50`, LSTM `0.5878`, delta `+0.0545`
- tick `60442`, seconds `75.50`, LSTM `0.5307`, delta `-0.0516`
- tick `62394`, seconds `106.00`, LSTM `0.0861`, delta `-0.0436`
- tick `57882`, seconds `35.50`, LSTM `0.6457`, delta `-0.0365`

## Top 15 local ridge features

- `lag_03__T_place_SIDEHALL`: coefficient `-0.004894`, |coef| `0.004894`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.004046`, |coef| `0.004046`
- `lag_15__T_place_SIDEHALL`: coefficient `-0.003725`, |coef| `0.003725`
- `lag_00__kill_diff_last_3s`: coefficient `0.003238`, |coef| `0.003238`
- `lag_00__damage_diff_last_5s`: coefficient `0.003222`, |coef| `0.003222`
- `lag_05__T_place_SIDEHALL`: coefficient `0.003011`, |coef| `0.003011`
- `lag_09__T_place_SIDEHALL`: coefficient `-0.002920`, |coef| `0.002920`
- `lag_09__CT_place_SIDEENTRANCE`: coefficient `0.002843`, |coef| `0.002843`
- `lag_01__T_place_SIDEHALL`: coefficient `-0.002789`, |coef| `0.002789`
- `lag_00__T_place_SIDEHALL`: coefficient `-0.002726`, |coef| `0.002726`
- `lag_03__T_place_HOUSE`: coefficient `0.002709`, |coef| `0.002709`
- `lag_03__T_place_CTSPAWN`: coefficient `-0.002572`, |coef| `0.002572`
- `lag_00__T_kills_last_3s`: coefficient `-0.002480`, |coef| `0.002480`
- `lag_07__T_place_SIDEHALL`: coefficient `0.002437`, |coef| `0.002437`
- `lag_11__T5__is_walking`: coefficient `0.002376`, |coef| `0.002376`

## Top 10 utility ridge features

- `lag_00__CT4__molly`: coefficient `0.001781` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.001437` (raises CT win probability)
- `lag_12__T3__smoke`: coefficient `0.001355` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.001253` (raises CT win probability)
- `lag_08__T3__smoke`: coefficient `-0.001192` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.001169` (raises CT win probability)
- `lag_11__CT_B_site_active_smokes`: coefficient `0.001159` (raises CT win probability)
- `lag_09__CT3__flash`: coefficient `0.001143` (raises CT win probability)
- `lag_05__CT3__flash`: coefficient `-0.001042` (lowers CT win probability)
- `lag_00__CT_flash_inv`: coefficient `0.000997` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_03__T_place_SIDEHALL`: coefficient `-0.004894` (lowers CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.004046` (raises CT win probability)
- `lag_15__T_place_SIDEHALL`: coefficient `-0.003725` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003238` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003222` (raises CT win probability)
- `lag_05__T_place_SIDEHALL`: coefficient `0.003011` (raises CT win probability)
- `lag_09__T_place_SIDEHALL`: coefficient `-0.002920` (lowers CT win probability)
- `lag_09__CT_place_SIDEENTRANCE`: coefficient `0.002843` (raises CT win probability)
- `lag_01__T_place_SIDEHALL`: coefficient `-0.002789` (lowers CT win probability)
- `lag_00__T_place_SIDEHALL`: coefficient `-0.002726` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `62362`, seconds `105.50`, LSTM delta `-0.3798`

Top all feature movements:
- `lag_15__T_place_SIDEHALL`: contribution `-0.024144`
- `lag_05__T_place_SIDEHALL`: contribution `-0.019515`
- `lag_07__T_place_SIDEHALL`: contribution `-0.015796`
- `lag_04__T_place_SIDEHALL`: contribution `-0.012936`
- `lag_03__T_place_CTSPAWN`: contribution `-0.012267`

Top utility-only movements:
- `lag_00__CT4__molly`: contribution `-0.004386`

### tick `62234`, seconds `103.50`, LSTM delta `+0.3003`

Top all feature movements:
- `lag_03__T_place_SIDEHALL`: contribution `+0.031718`
- `lag_01__T_place_SIDEHALL`: contribution `+0.018078`
- `lag_00__T_place_SIDEHALL`: contribution `+0.017666`
- `lag_07__T_place_SIDEHALL`: contribution `+0.015796`
- `lag_09__CT_place_SIDEENTRANCE`: contribution `+0.011443`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62074`, seconds `101.00`, LSTM delta `-0.1509`

Top all feature movements:
- `lag_07__T_place_SIDEHALL`: contribution `+0.015796`
- `lag_02__T_place_SIDEHALL`: contribution `-0.012423`
- `lag_06__T_place_SIDEHALL`: contribution `-0.011185`
- `lag_00__T_kills_last_3s`: contribution `-0.007858`
- `lag_00__kill_diff_last_3s`: contribution `-0.007793`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62106`, seconds `101.50`, LSTM delta `-0.1163`

Top all feature movements:
- `lag_03__T_place_SIDEHALL`: contribution `-0.031718`
- `lag_07__T_place_SIDEHALL`: contribution `+0.015796`
- `lag_08__T_place_SIDEHALL`: contribution `-0.015207`
- `lag_10__CT_place_HOUSE`: contribution `+0.005839`
- `lag_05__CT_place_SIDEENTRANCE`: contribution `-0.005624`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `62138`, seconds `102.00`, LSTM delta `-0.0677`

Top all feature movements:
- `lag_09__T_place_SIDEHALL`: contribution `-0.018927`
- `lag_00__T_place_SIDEHALL`: contribution `+0.017666`
- `lag_08__T_place_SIDEHALL`: contribution `-0.015207`
- `lag_04__T_place_SIDEHALL`: contribution `+0.012936`
- `lag_12__CT_place_HOUSE`: contribution `+0.004804`

Top utility-only movements:
- No utility movement among the top local contributors.
