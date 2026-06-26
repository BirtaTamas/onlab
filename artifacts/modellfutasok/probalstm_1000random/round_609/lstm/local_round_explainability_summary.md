# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `19772`, seconds `15.50`, LSTM `0.0401`, delta `-0.0635`
- tick `18812`, seconds `0.50`, LSTM `0.0765`, delta `-0.0617`
- tick `19004`, seconds `3.50`, LSTM `0.0884`, delta `+0.0247`
- tick `19100`, seconds `5.00`, LSTM `0.1089`, delta `+0.0224`
- tick `19804`, seconds `16.00`, LSTM `0.0182`, delta `-0.0219`
- tick `19132`, seconds `5.50`, LSTM `0.1284`, delta `+0.0196`
- tick `19612`, seconds `13.00`, LSTM `0.0992`, delta `-0.0186`
- tick `20764`, seconds `31.00`, LSTM `0.0081`, delta `-0.0168`
- tick `18844`, seconds `1.00`, LSTM `0.0618`, delta `-0.0147`
- tick `19516`, seconds `11.50`, LSTM `0.1174`, delta `-0.0121`

## Top 15 local ridge features

- `lag_01__T_place_TSPAWN`: coefficient `-0.000592`, |coef| `0.000592`
- `lag_02__CT_place_HOLE`: coefficient `-0.000591`, |coef| `0.000591`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000569`, |coef| `0.000569`
- `lag_00__CT_velocity_mean`: coefficient `-0.000489`, |coef| `0.000489`
- `lag_00__CT_place_MIDDOORS`: coefficient `0.000429`, |coef| `0.000429`
- `lag_01__utility_inv_diff`: coefficient `0.000423`, |coef| `0.000423`
- `lag_00__T_velocity_mean`: coefficient `-0.000416`, |coef| `0.000416`
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `0.000364`, |coef| `0.000364`
- `lag_01__molly_inv_diff`: coefficient `0.000353`, |coef| `0.000353`
- `lag_00__T_kills_last_3s`: coefficient `-0.000347`, |coef| `0.000347`
- `lag_01__flash_inv_diff`: coefficient `0.000347`, |coef| `0.000347`
- `lag_00__T2__smoke`: coefficient `0.000347`, |coef| `0.000347`
- `lag_01__CT3__duck_amount`: coefficient `-0.000340`, |coef| `0.000340`
- `lag_01__T4__has_bomb`: coefficient `-0.000332`, |coef| `0.000332`
- `lag_01__T1__utility_total`: coefficient `-0.000330`, |coef| `0.000330`

## Top 10 utility ridge features

- `lag_01__utility_inv_diff`: coefficient `0.000423` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000353` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000347` (raises CT win probability)
- `lag_00__T2__smoke`: coefficient `0.000347` (raises CT win probability)
- `lag_01__T1__utility_total`: coefficient `-0.000330` (lowers CT win probability)
- `lag_01__smoke_inv_diff`: coefficient `0.000324` (raises CT win probability)
- `lag_01__T_utility_inv`: coefficient `-0.000305` (lowers CT win probability)
- `lag_01__T1__flash`: coefficient `-0.000300` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000299` (lowers CT win probability)
- `lag_01__T_flash_inv`: coefficient `-0.000282` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__T_place_TSPAWN`: coefficient `-0.000592` (lowers CT win probability)
- `lag_02__CT_place_HOLE`: coefficient `-0.000591` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000569` (lowers CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.000489` (lowers CT win probability)
- `lag_00__CT_place_MIDDOORS`: coefficient `0.000429` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000416` (lowers CT win probability)
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `0.000364` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000347` (lowers CT win probability)
- `lag_01__CT3__duck_amount`: coefficient `-0.000340` (lowers CT win probability)
- `lag_01__T4__has_bomb`: coefficient `-0.000332` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `19772`, seconds `15.50`, LSTM delta `-0.0635`

Top all feature movements:
- `lag_02__CT_place_HOLE`: contribution `-0.006598`
- `lag_05__CT_place_SHORTSTAIRS`: contribution `-0.002028`
- `lag_02__CT_place_BDOORS`: contribution `-0.001525`
- `lag_05__CT_place_BDOORS`: contribution `-0.001497`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `-0.001457`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `18812`, seconds `0.50`, LSTM delta `-0.0617`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002722`
- `lag_01__T_place_TSPAWN`: contribution `-0.002620`
- `lag_01__utility_inv_diff`: contribution `-0.001394`
- `lag_00__CT_velocity_mean`: contribution `-0.001385`
- `lag_00__T_velocity_mean`: contribution `-0.001331`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.001394`
- `lag_01__molly_inv_diff`: contribution `-0.000984`
- `lag_01__flash_inv_diff`: contribution `-0.000928`
- `lag_01__smoke_inv_diff`: contribution `-0.000824`
- `lag_01__T_utility_inv`: contribution `-0.000787`

### tick `19004`, seconds `3.50`, LSTM delta `+0.0247`

Top all feature movements:
- `lag_00__CT_place_MIDDOORS`: contribution `+0.002474`
- `lag_03__CT_place_UNDERA`: contribution `+0.001184`
- `lag_00__CT_macro_MID`: contribution `+0.000890`
- `lag_07__T_place_TSPAWN`: contribution `+0.000840`
- `lag_02__CT3__duck_amount`: contribution `+0.000805`

Top utility-only movements:
- `lag_07__T2__flash`: contribution `+0.000248`

### tick `19100`, seconds `5.00`, LSTM delta `+0.0224`

Top all feature movements:
- `lag_01__CT3__duck_amount`: contribution `+0.001265`
- `lag_03__CT_place_MIDDOORS`: contribution `+0.001126`
- `lag_06__CT_place_UNDERA`: contribution `+0.001099`
- `lag_01__CT_place_EXTENDEDA`: contribution `+0.000907`
- `lag_00__T2__smoke`: contribution `+0.000762`

Top utility-only movements:
- `lag_00__T2__smoke`: contribution `+0.000762`

### tick `19804`, seconds `16.00`, LSTM delta `-0.0219`

Top all feature movements:
- `lag_03__CT_place_HOLE`: contribution `-0.003517`
- `lag_00__CT_place_HOLE`: contribution `-0.002125`
- `lag_00__CT_place_MIDDOORS`: contribution `-0.001237`
- `lag_00__T_kills_last_3s`: contribution `-0.001101`
- `lag_06__CT_place_SHORTSTAIRS`: contribution `-0.001018`

Top utility-only movements:
- No utility movement among the top local contributors.
