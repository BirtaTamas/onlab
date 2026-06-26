# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `19`

## Largest probability jumps

- tick `154411`, seconds `116.00`, LSTM `0.5569`, delta `+0.3350`
- tick `154635`, seconds `119.50`, LSTM `0.6780`, delta `+0.3098`
- tick `149867`, seconds `45.00`, LSTM `0.6717`, delta `+0.2573`
- tick `154571`, seconds `118.50`, LSTM `0.4044`, delta `-0.2566`
- tick `151115`, seconds `64.50`, LSTM `0.4510`, delta `+0.2361`
- tick `154699`, seconds `120.50`, LSTM `0.8758`, delta `+0.1907`
- tick `150955`, seconds `62.00`, LSTM `0.3412`, delta `-0.1670`
- tick `147819`, seconds `13.00`, LSTM `0.3321`, delta `-0.1419`
- tick `151467`, seconds `70.00`, LSTM `0.5280`, delta `+0.0946`
- tick `149835`, seconds `44.50`, LSTM `0.4145`, delta `+0.0848`

## Top 15 local ridge features

- `lag_01__CT_place_HUT`: coefficient `-0.006788`, |coef| `0.006788`
- `lag_06__CT_place_HUT`: coefficient `0.006579`, |coef| `0.006579`
- `lag_00__kill_diff_last_3s`: coefficient `0.005415`, |coef| `0.005415`
- `lag_00__T_shots_fired_sum`: coefficient `-0.004932`, |coef| `0.004932`
- `lag_00__CT_kills_last_3s`: coefficient `0.004920`, |coef| `0.004920`
- `lag_05__CT_place_SECRET`: coefficient `-0.004274`, |coef| `0.004274`
- `lag_13__CT_place_HUT`: coefficient `0.003443`, |coef| `0.003443`
- `lag_00__damage_diff_last_5s`: coefficient `0.003399`, |coef| `0.003399`
- `lag_00__CT_defusing_count`: coefficient `0.002870`, |coef| `0.002870`
- `lag_02__CT_place_HUT`: coefficient `-0.002832`, |coef| `0.002832`
- `lag_11__CT_place_DECON`: coefficient `0.002815`, |coef| `0.002815`
- `lag_01__CT_shots_fired_sum`: coefficient `0.002591`, |coef| `0.002591`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.002554`, |coef| `0.002554`
- `lag_00__T_duck_amount_mean`: coefficient `-0.002547`, |coef| `0.002547`
- `lag_06__CT_place_LOBBY`: coefficient `-0.002470`, |coef| `0.002470`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002554` (lowers CT win probability)
- `lag_11__T_flashes_last_5s`: coefficient `0.002200` (raises CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.001785` (lowers CT win probability)
- `lag_12__T_flashes_last_5s`: coefficient `0.001782` (raises CT win probability)
- `lag_13__T_flashes_last_5s`: coefficient `0.001274` (raises CT win probability)
- `lag_00__T3__flash`: coefficient `-0.001241` (lowers CT win probability)
- `lag_02__CT3__smoke`: coefficient `0.001107` (raises CT win probability)
- `lag_00__flash_inv_diff`: coefficient `0.001047` (raises CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.001030` (lowers CT win probability)
- `lag_07__T1__flash_duration`: coefficient `-0.001012` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_HUT`: coefficient `-0.006788` (lowers CT win probability)
- `lag_06__CT_place_HUT`: coefficient `0.006579` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.005415` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.004932` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004920` (raises CT win probability)
- `lag_05__CT_place_SECRET`: coefficient `-0.004274` (lowers CT win probability)
- `lag_13__CT_place_HUT`: coefficient `0.003443` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003399` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002870` (raises CT win probability)
- `lag_02__CT_place_HUT`: coefficient `-0.002832` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `154411`, seconds `116.00`, LSTM delta `+0.3350`

Top all feature movements:
- `lag_01__CT_place_HUT`: contribution `+0.066196`
- `lag_06__CT_place_HUT`: contribution `+0.064160`
- `lag_10__CT_place_HUT`: contribution `+0.020601`
- `lag_06__CT_place_LOBBY`: contribution `+0.020221`
- `lag_10__CT_place_LOBBY`: contribution `+0.019813`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `154635`, seconds `119.50`, LSTM delta `+0.3098`

Top all feature movements:
- `lag_00__T_shots_fired_sum`: contribution `+0.040672`
- `lag_13__CT_place_HUT`: contribution `+0.033574`
- `lag_13__CT_place_LOBBY`: contribution `+0.017469`
- `lag_08__CT_place_HUT`: contribution `+0.016551`
- `lag_00__T_flash_alpha_mean`: contribution `+0.015493`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.015493`
- `lag_00__T3__flash`: contribution `+0.003657`

### tick `149867`, seconds `45.00`, LSTM delta `+0.2573`

Top all feature movements:
- `lag_11__CT_place_DECON`: contribution `+0.044755`
- `lag_04__T_place_GARAGE`: contribution `+0.021755`
- `lag_00__T_place_GARAGE`: contribution `+0.015491`
- `lag_00__T_shots_fired_sum`: contribution `+0.014790`
- `lag_00__CT_kills_last_3s`: contribution `+0.014206`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `154571`, seconds `118.50`, LSTM delta `-0.2566`

Top all feature movements:
- `lag_06__CT_place_HUT`: contribution `-0.064160`
- `lag_11__CT_place_LOBBY`: contribution `-0.018676`
- `lag_00__T_shots_fired_sum`: contribution `-0.018487`
- `lag_00__T_duck_amount_mean`: contribution `-0.014812`
- `lag_00__kill_diff_last_3s`: contribution `-0.013033`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `151115`, seconds `64.50`, LSTM delta `+0.2361`

Top all feature movements:
- `lag_05__CT_place_SECRET`: contribution `+0.043996`
- `lag_10__CT_place_SECRET`: contribution `+0.017673`
- `lag_00__CT_kills_last_3s`: contribution `+0.014206`
- `lag_00__kill_diff_last_3s`: contribution `+0.013033`
- `lag_02__T3__duck_amount`: contribution `-0.007577`

Top utility-only movements:
- No utility movement among the top local contributors.
