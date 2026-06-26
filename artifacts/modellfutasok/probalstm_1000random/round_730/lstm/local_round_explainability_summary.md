# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `13`

## Largest probability jumps

- tick `118239`, seconds `64.50`, LSTM `0.3899`, delta `-0.3780`
- tick `118399`, seconds `67.00`, LSTM `0.5786`, delta `+0.3694`
- tick `116671`, seconds `40.00`, LSTM `0.8197`, delta `+0.2369`
- tick `115039`, seconds `14.50`, LSTM `0.6214`, delta `-0.2220`
- tick `115007`, seconds `14.00`, LSTM `0.8434`, delta `+0.2089`
- tick `117567`, seconds `54.00`, LSTM `0.6220`, delta `-0.1792`
- tick `114943`, seconds `13.00`, LSTM `0.6562`, delta `+0.1430`
- tick `117823`, seconds `58.00`, LSTM `0.7416`, delta `+0.1288`
- tick `118271`, seconds `65.00`, LSTM `0.2619`, delta `-0.1280`
- tick `117215`, seconds `48.50`, LSTM `0.8221`, delta `-0.1079`

## Top 15 local ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007504`, |coef| `0.007504`
- `lag_00__T_place_DECON`: coefficient `-0.004713`, |coef| `0.004713`
- `lag_13__CT_defusing_count`: coefficient `-0.004302`, |coef| `0.004302`
- `lag_15__T_place_DECON`: coefficient `0.003607`, |coef| `0.003607`
- `lag_02__CT_place_DECON`: coefficient `0.003299`, |coef| `0.003299`
- `lag_00__kill_diff_last_3s`: coefficient `0.003175`, |coef| `0.003175`
- `lag_07__T_place_DECON`: coefficient `-0.002991`, |coef| `0.002991`
- `lag_05__CT5__duck_amount`: coefficient `-0.002741`, |coef| `0.002741`
- `lag_00__damage_diff_last_5s`: coefficient `0.002583`, |coef| `0.002583`
- `lag_03__T_duck_amount_mean`: coefficient `-0.002472`, |coef| `0.002472`
- `lag_01__CT_defusing_count`: coefficient `0.002468`, |coef| `0.002468`
- `lag_00__CT_kills_last_3s`: coefficient `0.002433`, |coef| `0.002433`
- `lag_00__CT_duck_amount_mean`: coefficient `0.002363`, |coef| `0.002363`
- `lag_02__T_duck_amount_mean`: coefficient `-0.002337`, |coef| `0.002337`
- `lag_12__T1__is_walking`: coefficient `0.002237`, |coef| `0.002237`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.002001` (lowers CT win probability)
- `lag_15__T_flash_alpha_mean`: coefficient `-0.001118` (lowers CT win probability)
- `lag_01__CT4__flash_duration`: coefficient `0.000754` (raises CT win probability)
- `lag_08__T_flash_alpha_mean`: coefficient `0.000748` (raises CT win probability)
- `lag_09__T_flash_alpha_mean`: coefficient `0.000744` (raises CT win probability)
- `lag_12__T_flash_alpha_mean`: coefficient `-0.000684` (lowers CT win probability)
- `lag_02__CT4__flash_duration`: coefficient `0.000682` (raises CT win probability)
- `lag_07__T_flash_alpha_mean`: coefficient `0.000674` (raises CT win probability)
- `lag_06__T_flash_alpha_mean`: coefficient `0.000617` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.000581` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_defusing_count`: coefficient `0.007504` (raises CT win probability)
- `lag_00__T_place_DECON`: coefficient `-0.004713` (lowers CT win probability)
- `lag_13__CT_defusing_count`: coefficient `-0.004302` (lowers CT win probability)
- `lag_15__T_place_DECON`: coefficient `0.003607` (raises CT win probability)
- `lag_02__CT_place_DECON`: coefficient `0.003299` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003175` (raises CT win probability)
- `lag_07__T_place_DECON`: coefficient `-0.002991` (lowers CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `-0.002741` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002583` (raises CT win probability)
- `lag_03__T_duck_amount_mean`: coefficient `-0.002472` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `118239`, seconds `64.50`, LSTM delta `-0.3780`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `-0.072746`
- `lag_15__T_place_DECON`: contribution `-0.057953`
- `lag_07__T_place_DECON`: contribution `-0.048050`
- `lag_13__CT_defusing_count`: contribution `-0.041708`
- `lag_00__kill_diff_last_3s`: contribution `-0.007643`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118399`, seconds `67.00`, LSTM delta `+0.3694`

Top all feature movements:
- `lag_00__T_place_DECON`: contribution `+0.075713`
- `lag_02__CT_place_DECON`: contribution `+0.052449`
- `lag_12__T_place_DECON`: contribution `+0.025693`
- `lag_05__CT_defusing_count`: contribution `+0.018953`
- `lag_00__T_flash_alpha_mean`: contribution `+0.012138`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.012138`

### tick `116671`, seconds `40.00`, LSTM delta `+0.2369`

Top all feature movements:
- `lag_05__CT_place_DECON`: contribution `+0.029731`
- `lag_14__CT_place_VENTS`: contribution `+0.016780`
- `lag_15__CT_place_ADMIN`: contribution `+0.012657`
- `lag_14__CT_place_ADMIN`: contribution `+0.010951`
- `lag_01__CT_place_DECON`: contribution `+0.010787`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115039`, seconds `14.50`, LSTM delta `-0.2220`

Top all feature movements:
- `lag_09__T_place_SQUEAKY`: contribution `-0.016110`
- `lag_05__T_place_HUT`: contribution `-0.013302`
- `lag_15__CT_place_ADMIN`: contribution `-0.012657`
- `lag_00__kill_diff_last_3s`: contribution `-0.007643`
- `lag_15__CT_place_HELL`: contribution `-0.005882`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115007`, seconds `14.00`, LSTM delta `+0.2089`

Top all feature movements:
- `lag_08__T_place_SQUEAKY`: contribution `+0.019107`
- `lag_14__CT_place_ADMIN`: contribution `-0.010951`
- `lag_14__CT_place_HELL`: contribution `+0.009886`
- `lag_04__T_place_HUT`: contribution `+0.009684`
- `lag_11__CT_place_ADMIN`: contribution `+0.009326`

Top utility-only movements:
- No utility movement among the top local contributors.
