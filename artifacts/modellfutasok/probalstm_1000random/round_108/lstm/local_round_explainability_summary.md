# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `10`

## Largest probability jumps

- tick `59563`, seconds `4.00`, LSTM `0.0617`, delta `-0.0798`
- tick `59339`, seconds `0.50`, LSTM `0.1039`, delta `-0.0560`
- tick `60395`, seconds `17.00`, LSTM `0.0246`, delta `-0.0490`
- tick `60235`, seconds `14.50`, LSTM `0.0934`, delta `+0.0290`
- tick `59499`, seconds `3.00`, LSTM `0.1239`, delta `+0.0270`
- tick `59371`, seconds `1.00`, LSTM `0.0853`, delta `-0.0186`
- tick `59531`, seconds `3.50`, LSTM `0.1414`, delta `+0.0175`
- tick `62699`, seconds `53.00`, LSTM `0.0083`, delta `-0.0120`
- tick `59691`, seconds `6.00`, LSTM `0.0324`, delta `-0.0115`
- tick `60331`, seconds `16.00`, LSTM `0.0748`, delta `-0.0112`

## Top 15 local ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000766`, |coef| `0.000766`
- `lag_07__CT_flashes_last_5s`: coefficient `-0.000612`, |coef| `0.000612`
- `lag_00__T1__is_scoped`: coefficient `0.000486`, |coef| `0.000486`
- `lag_00__CT_place_MIDDOORS`: coefficient `0.000485`, |coef| `0.000485`
- `lag_02__CT4__duck_amount`: coefficient `0.000432`, |coef| `0.000432`
- `lag_00__T_velocity_mean`: coefficient `-0.000421`, |coef| `0.000421`
- `lag_02__CT_place_MIDDOORS`: coefficient `-0.000420`, |coef| `0.000420`
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000405`, |coef| `0.000405`
- `lag_01__T_place_TSPAWN`: coefficient `-0.000378`, |coef| `0.000378`
- `lag_09__T_he_last_5s`: coefficient `0.000378`, |coef| `0.000378`
- `lag_01__CT_place_UPPERTUNNEL`: coefficient `0.000367`, |coef| `0.000367`
- `lag_00__T_place_OUTSIDETUNNEL`: coefficient `-0.000366`, |coef| `0.000366`
- `lag_00__T_kills_last_3s`: coefficient `-0.000363`, |coef| `0.000363`
- `lag_08__CT_place_CTSPAWN`: coefficient `-0.000330`, |coef| `0.000330`
- `lag_01__CT_flashes_last_5s`: coefficient `-0.000329`, |coef| `0.000329`

## Top 10 utility ridge features

- `lag_00__CT_flashes_last_5s`: coefficient `-0.000766` (lowers CT win probability)
- `lag_07__CT_flashes_last_5s`: coefficient `-0.000612` (lowers CT win probability)
- `lag_09__T_he_last_5s`: coefficient `0.000378` (raises CT win probability)
- `lag_01__CT_flashes_last_5s`: coefficient `-0.000329` (lowers CT win probability)
- `lag_01__utility_inv_diff`: coefficient `0.000323` (raises CT win probability)
- `lag_08__utility_inv_diff`: coefficient `0.000299` (raises CT win probability)
- `lag_05__CT_flashes_last_5s`: coefficient `0.000292` (raises CT win probability)
- `lag_01__molly_inv_diff`: coefficient `0.000290` (raises CT win probability)
- `lag_01__flash_inv_diff`: coefficient `0.000280` (raises CT win probability)
- `lag_08__flash_inv_diff`: coefficient `0.000270` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T1__is_scoped`: coefficient `0.000486` (raises CT win probability)
- `lag_00__CT_place_MIDDOORS`: coefficient `0.000485` (raises CT win probability)
- `lag_02__CT4__duck_amount`: coefficient `0.000432` (raises CT win probability)
- `lag_00__T_velocity_mean`: coefficient `-0.000421` (lowers CT win probability)
- `lag_02__CT_place_MIDDOORS`: coefficient `-0.000420` (lowers CT win probability)
- `lag_01__CT_place_CTSPAWN`: coefficient `-0.000405` (lowers CT win probability)
- `lag_01__T_place_TSPAWN`: coefficient `-0.000378` (lowers CT win probability)
- `lag_01__CT_place_UPPERTUNNEL`: coefficient `0.000367` (raises CT win probability)
- `lag_00__T_place_OUTSIDETUNNEL`: coefficient `-0.000366` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000363` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `59563`, seconds `4.00`, LSTM delta `-0.0798`

Top all feature movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.006731`
- `lag_00__T1__is_scoped`: contribution `-0.002776`
- `lag_02__CT_place_MIDDOORS`: contribution `-0.002423`
- `lag_00__T_place_OUTSIDETUNNEL`: contribution `-0.001831`
- `lag_02__CT4__duck_amount`: contribution `-0.001585`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `-0.006731`
- `lag_08__utility_inv_diff`: contribution `-0.000987`
- `lag_08__flash_inv_diff`: contribution `-0.000835`

### tick `59339`, seconds `0.50`, LSTM delta `-0.0560`

Top all feature movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.008423`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.001936`
- `lag_01__T_place_TSPAWN`: contribution `-0.001675`
- `lag_00__T_velocity_mean`: contribution `-0.001544`
- `lag_01__utility_inv_diff`: contribution `-0.001067`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.008423`
- `lag_01__utility_inv_diff`: contribution `-0.001067`
- `lag_01__flash_inv_diff`: contribution `-0.000867`
- `lag_01__molly_inv_diff`: contribution `-0.000809`
- `lag_01__T_utility_inv`: contribution `-0.000622`

### tick `60395`, seconds `17.00`, LSTM delta `-0.0490`

Top all feature movements:
- `lag_09__T_he_last_5s`: contribution `-0.004931`
- `lag_05__CT_place_TUNNELSTAIRS`: contribution `-0.004123`
- `lag_14__CT_place_TUNNELSTAIRS`: contribution `-0.003522`
- `lag_01__CT_place_TUNNELSTAIRS`: contribution `-0.002996`
- `lag_01__CT_place_UPPERTUNNEL`: contribution `-0.002812`

Top utility-only movements:
- `lag_09__T_he_last_5s`: contribution `-0.004931`

### tick `60235`, seconds `14.50`, LSTM delta `+0.0290`

Top all feature movements:
- `lag_14__T_he_last_5s`: contribution `+0.002973`
- `lag_00__T1__is_scoped`: contribution `+0.002776`
- `lag_04__T_he_last_5s`: contribution `+0.002440`
- `lag_11__CT_place_TUNNELSTAIRS`: contribution `+0.002394`
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `+0.001883`

Top utility-only movements:
- `lag_14__T_he_last_5s`: contribution `+0.002973`
- `lag_04__T_he_last_5s`: contribution `+0.002440`

### tick `59499`, seconds `3.00`, LSTM delta `+0.0270`

Top all feature movements:
- `lag_05__CT_flashes_last_5s`: contribution `+0.003216`
- `lag_00__CT_place_MIDDOORS`: contribution `+0.002800`
- `lag_00__T1__is_scoped`: contribution `+0.002776`
- `lag_06__CT_place_CTSPAWN`: contribution `+0.001130`
- `lag_00__CT_macro_MID`: contribution `+0.001073`

Top utility-only movements:
- `lag_05__CT_flashes_last_5s`: contribution `+0.003216`
