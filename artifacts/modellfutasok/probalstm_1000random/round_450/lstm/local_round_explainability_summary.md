# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `19832`, seconds `56.50`, LSTM `0.5829`, delta `+0.4000`
- tick `20920`, seconds `73.50`, LSTM `0.5385`, delta `-0.2496`
- tick `20440`, seconds `66.00`, LSTM `0.8515`, delta `+0.2294`
- tick `19736`, seconds `55.00`, LSTM `0.5030`, delta `-0.2186`
- tick `19768`, seconds `55.50`, LSTM `0.3213`, delta `-0.1817`
- tick `18232`, seconds `31.50`, LSTM `0.6858`, delta `-0.1717`
- tick `19480`, seconds `51.00`, LSTM `0.7607`, delta `+0.1513`
- tick `19800`, seconds `56.00`, LSTM `0.1829`, delta `-0.1384`
- tick `20632`, seconds `69.00`, LSTM `0.8444`, delta `-0.1139`
- tick `20152`, seconds `61.50`, LSTM `0.5460`, delta `-0.1044`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004255`, |coef| `0.004255`
- `lag_00__damage_diff_last_5s`: coefficient `0.003998`, |coef| `0.003998`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003613`, |coef| `0.003613`
- `lag_00__T_place_UNDERPASS`: coefficient `-0.003606`, |coef| `0.003606`
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `-0.003365`, |coef| `0.003365`
- `lag_15__T_place_LADDER`: coefficient `0.003296`, |coef| `0.003296`
- `lag_00__CT_place_SHOP`: coefficient `0.003093`, |coef| `0.003093`
- `lag_15__T_place_SNIPERSNEST`: coefficient `0.002907`, |coef| `0.002907`
- `lag_02__CT2__is_walking`: coefficient `-0.002805`, |coef| `0.002805`
- `lag_00__CT_kills_last_3s`: coefficient `0.002801`, |coef| `0.002801`
- `lag_04__T_place_LADDER`: coefficient `0.002761`, |coef| `0.002761`
- `lag_09__T4__duck_amount`: coefficient `-0.002688`, |coef| `0.002688`
- `lag_06__CT2__duck_amount`: coefficient `0.002667`, |coef| `0.002667`
- `lag_08__T5__duck_amount`: coefficient `-0.002627`, |coef| `0.002627`
- `lag_15__T1__duck_amount`: coefficient `-0.002572`, |coef| `0.002572`

## Top 10 utility ridge features

- `lag_00__T5__flash_duration`: coefficient `-0.001731` (lowers CT win probability)
- `lag_00__CT_flash_alpha_mean`: coefficient `0.001587` (raises CT win probability)
- `lag_11__CT_flash_alpha_mean`: coefficient `0.001582` (raises CT win probability)
- `lag_13__CT5__smoke`: coefficient `0.001360` (raises CT win probability)
- `lag_09__CT_flash_alpha_mean`: coefficient `0.001355` (raises CT win probability)
- `lag_01__CT_A_site_active_smokes`: coefficient `0.001329` (raises CT win probability)
- `lag_11__CT5__smoke`: coefficient `-0.001328` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.001305` (raises CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001227` (lowers CT win probability)
- `lag_08__CT_flash_alpha_mean`: coefficient `0.001175` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004255` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003998` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.003613` (lowers CT win probability)
- `lag_00__T_place_UNDERPASS`: coefficient `-0.003606` (lowers CT win probability)
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `-0.003365` (lowers CT win probability)
- `lag_15__T_place_LADDER`: coefficient `0.003296` (raises CT win probability)
- `lag_00__CT_place_SHOP`: coefficient `0.003093` (raises CT win probability)
- `lag_15__T_place_SNIPERSNEST`: coefficient `0.002907` (raises CT win probability)
- `lag_02__CT2__is_walking`: coefficient `-0.002805` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002801` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `19832`, seconds `56.50`, LSTM delta `+0.4000`

Top all feature movements:
- `lag_00__T_place_UNDERPASS`: contribution `+0.014126`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `+0.013712`
- `lag_03__CT_place_SHOP`: contribution `+0.012852`
- `lag_00__kill_diff_last_3s`: contribution `+0.010242`
- `lag_06__CT2__duck_amount`: contribution `+0.010160`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20920`, seconds `73.50`, LSTM delta `-0.2496`

Top all feature movements:
- `lag_15__T_place_LADDER`: contribution `-0.074505`
- `lag_00__CT_place_SHOP`: contribution `-0.015516`
- `lag_13__CT_shots_fired_sum`: contribution `-0.014035`
- `lag_00__kill_diff_last_3s`: contribution `-0.010242`
- `lag_00__T_duck_amount_mean`: contribution `-0.008464`

Top utility-only movements:
- `lag_00__CT_flash_alpha_mean`: contribution `-0.006094`
- `lag_00__CT4__flash`: contribution `-0.002940`

### tick `20440`, seconds `66.00`, LSTM delta `+0.2294`

Top all feature movements:
- `lag_04__T_place_LADDER`: contribution `+0.062423`
- `lag_00__T_place_LADDER`: contribution `+0.033473`
- `lag_04__T_place_SNIPERSNEST`: contribution `+0.025526`
- `lag_10__T_place_SNIPERSNEST`: contribution `+0.021423`
- `lag_00__kill_diff_last_3s`: contribution `+0.010242`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19736`, seconds `55.00`, LSTM delta `-0.2186`

Top all feature movements:
- `lag_00__CT_place_SHOP`: contribution `-0.015516`
- `lag_00__T_place_UNDERPASS`: contribution `-0.014126`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `-0.013712`
- `lag_00__kill_diff_last_3s`: contribution `-0.010242`
- `lag_06__CT2__duck_amount`: contribution `+0.008432`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `19768`, seconds `55.50`, LSTM delta `-0.1817`

Top all feature movements:
- `lag_15__T1__duck_amount`: contribution `-0.010070`
- `lag_01__CT_place_SHOP`: contribution `-0.009553`
- `lag_09__T4__duck_amount`: contribution `-0.008589`
- `lag_06__CT2__duck_amount`: contribution `-0.008432`
- `lag_13__T4__duck_amount`: contribution `-0.007645`

Top utility-only movements:
- No utility movement among the top local contributors.
