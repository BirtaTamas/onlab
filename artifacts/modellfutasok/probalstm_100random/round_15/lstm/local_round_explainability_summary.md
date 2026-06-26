# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `22`

## Largest probability jumps

- tick `185141`, seconds `117.00`, LSTM `0.8584`, delta `+0.4479`
- tick `185045`, seconds `115.50`, LSTM `0.4810`, delta `-0.3193`
- tick `184597`, seconds `108.50`, LSTM `0.8023`, delta `+0.3190`
- tick `185237`, seconds `118.50`, LSTM `0.9184`, delta `+0.1595`
- tick `185525`, seconds `123.00`, LSTM `0.7763`, delta `-0.1452`
- tick `185173`, seconds `117.50`, LSTM `0.7378`, delta `-0.1206`
- tick `180757`, seconds `48.50`, LSTM `0.4036`, delta `-0.0961`
- tick `182485`, seconds `75.50`, LSTM `0.5186`, delta `+0.0922`
- tick `182453`, seconds `75.00`, LSTM `0.4264`, delta `+0.0778`
- tick `183797`, seconds `96.00`, LSTM `0.4949`, delta `-0.0541`

## Top 15 local ridge features

- `lag_02__CT_place_VENDING`: coefficient `-0.003989`, |coef| `0.003989`
- `lag_08__CT_place_SQUEAKY`: coefficient `-0.003490`, |coef| `0.003490`
- `lag_03__CT_place_SQUEAKY`: coefficient `-0.003438`, |coef| `0.003438`
- `lag_00__CT_defusing_count`: coefficient `0.002941`, |coef| `0.002941`
- `lag_02__CT_place_LOBBY`: coefficient `0.002691`, |coef| `0.002691`
- `lag_08__CT_place_LOBBY`: coefficient `0.002667`, |coef| `0.002667`
- `lag_00__kill_diff_last_3s`: coefficient `0.002574`, |coef| `0.002574`
- `lag_11__CT_place_VENTS`: coefficient `-0.002351`, |coef| `0.002351`
- `lag_00__CT_kills_last_3s`: coefficient `0.002247`, |coef| `0.002247`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001983`, |coef| `0.001983`
- `lag_08__CT_place_HEAVEN`: coefficient `0.001963`, |coef| `0.001963`
- `lag_00__damage_diff_last_5s`: coefficient `0.001840`, |coef| `0.001840`
- `lag_13__CT_place_SQUEAKY`: coefficient `0.001753`, |coef| `0.001753`
- `lag_07__CT_place_RAFTERS`: coefficient `-0.001703`, |coef| `0.001703`
- `lag_00__T_place_HUT`: coefficient `-0.001672`, |coef| `0.001672`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.001659` (lowers CT win probability)
- `lag_03__T_flash_alpha_mean`: coefficient `-0.001070` (lowers CT win probability)
- `lag_03__CT4__flash`: coefficient `-0.000979` (lowers CT win probability)
- `lag_07__CT_A_site_active_infernos`: coefficient `0.000947` (raises CT win probability)
- `lag_08__CT5__molly`: coefficient `-0.000924` (lowers CT win probability)
- `lag_00__T5__flash_duration`: coefficient `-0.000869` (lowers CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000818` (raises CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `0.000770` (raises CT win probability)
- `lag_13__CT_A_site_active_infernos`: coefficient `-0.000769` (lowers CT win probability)
- `lag_02__T_flash_alpha_mean`: coefficient `-0.000730` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_VENDING`: coefficient `-0.003989` (lowers CT win probability)
- `lag_08__CT_place_SQUEAKY`: coefficient `-0.003490` (lowers CT win probability)
- `lag_03__CT_place_SQUEAKY`: coefficient `-0.003438` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.002941` (raises CT win probability)
- `lag_02__CT_place_LOBBY`: coefficient `0.002691` (raises CT win probability)
- `lag_08__CT_place_LOBBY`: coefficient `0.002667` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002574` (raises CT win probability)
- `lag_11__CT_place_VENTS`: coefficient `-0.002351` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002247` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001983` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `185141`, seconds `117.00`, LSTM delta `+0.4479`

Top all feature movements:
- `lag_08__CT_place_SQUEAKY`: contribution `+0.046410`
- `lag_03__CT_place_SQUEAKY`: contribution `+0.045725`
- `lag_08__CT_place_LOBBY`: contribution `+0.021832`
- `lag_11__CT_place_SQUEAKY`: contribution `+0.020373`
- `lag_11__CT_place_VENTS`: contribution `+0.019725`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.010066`

### tick `185045`, seconds `115.50`, LSTM delta `-0.3193`

Top all feature movements:
- `lag_08__CT_place_SQUEAKY`: contribution `-0.046410`
- `lag_03__CT_place_SQUEAKY`: contribution `-0.045725`
- `lag_08__CT_place_LOBBY`: contribution `-0.021832`
- `lag_00__CT_place_SQUEAKY`: contribution `-0.017553`
- `lag_07__T_place_HUT`: contribution `-0.012417`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `184597`, seconds `108.50`, LSTM delta `+0.3190`

Top all feature movements:
- `lag_02__CT_place_VENDING`: contribution `+0.068367`
- `lag_02__CT_place_LOBBY`: contribution `+0.022030`
- `lag_00__T_shots_fired_sum`: contribution `+0.011894`
- `lag_08__CT_place_HEAVEN`: contribution `+0.010601`
- `lag_06__T1__is_scoped`: contribution `+0.006981`

Top utility-only movements:
- `lag_07__CT_A_site_active_infernos`: contribution `+0.003343`

### tick `185237`, seconds `118.50`, LSTM delta `+0.1595`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.028506`
- `lag_11__CT_place_SQUEAKY`: contribution `-0.020373`
- `lag_14__CT_place_SQUEAKY`: contribution `+0.020040`
- `lag_09__CT_place_SQUEAKY`: contribution `+0.015587`
- `lag_06__CT_place_SQUEAKY`: contribution `-0.013778`

Top utility-only movements:
- `lag_03__T_flash_alpha_mean`: contribution `+0.006490`

### tick `185525`, seconds `123.00`, LSTM delta `-0.1452`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.028506`
- `lag_15__CT_place_SQUEAKY`: contribution `-0.018974`
- `lag_09__CT_defusing_count`: contribution `-0.011873`
- `lag_15__T1__is_scoped`: contribution `-0.004867`
- `lag_12__T_flash_alpha_mean`: contribution `-0.004671`

Top utility-only movements:
- `lag_12__T_flash_alpha_mean`: contribution `-0.004671`
- `lag_15__CT4__flash`: contribution `-0.001589`
