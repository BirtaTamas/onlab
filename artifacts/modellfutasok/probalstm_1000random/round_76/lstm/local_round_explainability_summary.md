# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `9`

## Largest probability jumps

- tick `56536`, seconds `80.50`, LSTM `0.2124`, delta `-0.4238`
- tick `57432`, seconds `94.50`, LSTM `0.6227`, delta `+0.3757`
- tick `55128`, seconds `58.50`, LSTM `0.5853`, delta `-0.2074`
- tick `58616`, seconds `113.00`, LSTM `0.5256`, delta `-0.1896`
- tick `57368`, seconds `93.50`, LSTM `0.2045`, delta `+0.1606`
- tick `54648`, seconds `51.00`, LSTM `0.6580`, delta `+0.1580`
- tick `56568`, seconds `81.00`, LSTM `0.1030`, delta `-0.1094`
- tick `58360`, seconds `109.00`, LSTM `0.7083`, delta `-0.0985`
- tick `57464`, seconds `95.00`, LSTM `0.7132`, delta `+0.0905`
- tick `54424`, seconds `47.50`, LSTM `0.5189`, delta `-0.0893`

## Top 15 local ridge features

- `lag_02__CT_place_ROOF`: coefficient `0.007937`, |coef| `0.007937`
- `lag_00__CT_place_SQUEAKY`: coefficient `0.002990`, |coef| `0.002990`
- `lag_09__T_place_HELL`: coefficient `-0.002947`, |coef| `0.002947`
- `lag_00__T_place_HEAVEN`: coefficient `-0.002705`, |coef| `0.002705`
- `lag_01__CT_place_ROOF`: coefficient `0.002612`, |coef| `0.002612`
- `lag_03__CT_place_ROOF`: coefficient `0.002435`, |coef| `0.002435`
- `lag_13__T_place_ADMIN`: coefficient `0.002252`, |coef| `0.002252`
- `lag_00__kill_diff_last_3s`: coefficient `0.002243`, |coef| `0.002243`
- `lag_06__CT_place_HEAVEN`: coefficient `-0.002099`, |coef| `0.002099`
- `lag_08__T_place_HELL`: coefficient `-0.002091`, |coef| `0.002091`
- `lag_00__T_place_RAFTERS`: coefficient `-0.001976`, |coef| `0.001976`
- `lag_13__T_place_HELL`: coefficient `-0.001966`, |coef| `0.001966`
- `lag_00__T_kills_last_3s`: coefficient `-0.001947`, |coef| `0.001947`
- `lag_03__T_place_RAFTERS`: coefficient `0.001933`, |coef| `0.001933`
- `lag_10__T_place_HEAVEN`: coefficient `0.001916`, |coef| `0.001916`

## Top 10 utility ridge features

- `lag_00__CT_mollies_last_5s`: coefficient `-0.000758` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `0.000501` (raises CT win probability)
- `lag_08__T2__flash_duration`: coefficient `-0.000480` (lowers CT win probability)
- `lag_01__CT_B_site_active_smokes`: coefficient `0.000469` (raises CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.000457` (raises CT win probability)
- `lag_09__T_A_site_active_infernos`: coefficient `-0.000446` (lowers CT win probability)
- `lag_09__T_B_site_active_infernos`: coefficient `-0.000424` (lowers CT win probability)
- `lag_13__T2__flash_duration`: coefficient `-0.000386` (lowers CT win probability)
- `lag_10__CT_mollies_last_5s`: coefficient `0.000383` (raises CT win probability)
- `lag_11__T1__molly`: coefficient `0.000379` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_ROOF`: coefficient `0.007937` (raises CT win probability)
- `lag_00__CT_place_SQUEAKY`: coefficient `0.002990` (raises CT win probability)
- `lag_09__T_place_HELL`: coefficient `-0.002947` (lowers CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.002705` (lowers CT win probability)
- `lag_01__CT_place_ROOF`: coefficient `0.002612` (raises CT win probability)
- `lag_03__CT_place_ROOF`: coefficient `0.002435` (raises CT win probability)
- `lag_13__T_place_ADMIN`: coefficient `0.002252` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002243` (raises CT win probability)
- `lag_06__CT_place_HEAVEN`: coefficient `-0.002099` (lowers CT win probability)
- `lag_08__T_place_HELL`: coefficient `-0.002091` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `56536`, seconds `80.50`, LSTM delta `-0.4238`

Top all feature movements:
- `lag_02__CT_place_ROOF`: contribution `-0.234127`
- `lag_06__CT_place_HEAVEN`: contribution `-0.011334`
- `lag_00__CT_place_HEAVEN`: contribution `-0.008968`
- `lag_06__CT_place_RAFTERS`: contribution `-0.008832`
- `lag_00__T_kills_last_3s`: contribution `-0.006167`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `57432`, seconds `94.50`, LSTM delta `+0.3757`

Top all feature movements:
- `lag_09__T_place_HELL`: contribution `+0.062833`
- `lag_03__T_place_RAFTERS`: contribution `+0.050582`
- `lag_00__CT_place_SQUEAKY`: contribution `+0.039769`
- `lag_02__T_place_RAFTERS`: contribution `+0.039480`
- `lag_00__T_place_HEAVEN`: contribution `+0.033192`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55128`, seconds `58.50`, LSTM delta `-0.2074`

Top all feature movements:
- `lag_09__T_place_HELL`: contribution `-0.062833`
- `lag_00__CT_place_SQUEAKY`: contribution `-0.039769`
- `lag_03__T_place_ADMIN`: contribution `-0.015713`
- `lag_00__CT_place_LOBBY`: contribution `-0.010774`
- `lag_02__T_place_GARAGE`: contribution `-0.008585`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `58616`, seconds `113.00`, LSTM delta `-0.1896`

Top all feature movements:
- `lag_08__T_place_DECON`: contribution `-0.025652`
- `lag_00__CT_place_DECON`: contribution `-0.022710`
- `lag_01__CT_place_DECON`: contribution `-0.021865`
- `lag_05__T_place_DECON`: contribution `-0.021214`
- `lag_12__T_place_VENTS`: contribution `-0.020003`

Top utility-only movements:
- `lag_09__T_A_site_active_infernos`: contribution `-0.001326`

### tick `57368`, seconds `93.50`, LSTM delta `+0.1606`

Top all feature movements:
- `lag_00__T_place_RAFTERS`: contribution `+0.051706`
- `lag_07__T_place_HELL`: contribution `+0.033052`
- `lag_01__T_place_HEAVEN`: contribution `+0.019253`
- `lag_08__T_place_HEAVEN`: contribution `+0.012403`
- `lag_01__T_place_RAFTERS`: contribution `+0.007665`

Top utility-only movements:
- No utility movement among the top local contributors.
