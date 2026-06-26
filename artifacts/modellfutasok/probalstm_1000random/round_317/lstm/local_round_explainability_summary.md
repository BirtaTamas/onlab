# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `10`

## Largest probability jumps

- tick `56223`, seconds `57.00`, LSTM `0.2837`, delta `-0.3841`
- tick `55359`, seconds `43.50`, LSTM `0.5562`, delta `+0.2236`
- tick `54751`, seconds `34.00`, LSTM `0.3750`, delta `-0.2200`
- tick `54463`, seconds `29.50`, LSTM `0.9266`, delta `+0.1499`
- tick `54111`, seconds `24.00`, LSTM `0.9171`, delta `+0.1150`
- tick `53727`, seconds `18.00`, LSTM `0.7800`, delta `+0.1028`
- tick `54559`, seconds `31.00`, LSTM `0.8514`, delta `-0.0947`
- tick `54591`, seconds `31.50`, LSTM `0.7591`, delta `-0.0923`
- tick `54623`, seconds `32.00`, LSTM `0.6844`, delta `-0.0746`
- tick `54719`, seconds `33.50`, LSTM `0.5950`, delta `-0.0730`

## Top 15 local ridge features

- `lag_02__CT_place_SHOP`: coefficient `0.004195`, |coef| `0.004195`
- `lag_00__kill_diff_last_3s`: coefficient `0.003753`, |coef| `0.003753`
- `lag_06__T_duck_amount_mean`: coefficient `-0.003363`, |coef| `0.003363`
- `lag_02__T4__is_scoped`: coefficient `-0.003299`, |coef| `0.003299`
- `lag_00__T_kills_last_3s`: coefficient `-0.003082`, |coef| `0.003082`
- `lag_00__damage_diff_last_5s`: coefficient `0.002730`, |coef| `0.002730`
- `lag_00__CT_place_SHOP`: coefficient `0.002496`, |coef| `0.002496`
- `lag_05__T_bomb_zone_count`: coefficient `-0.002485`, |coef| `0.002485`
- `lag_00__T_damage_last_5s`: coefficient `-0.002464`, |coef| `0.002464`
- `lag_03__CT_place_SHOP`: coefficient `0.002446`, |coef| `0.002446`
- `lag_03__T_duck_amount_mean`: coefficient `0.002408`, |coef| `0.002408`
- `lag_06__T4__duck_amount`: coefficient `-0.002386`, |coef| `0.002386`
- `lag_00__CT5__molly`: coefficient `0.002381`, |coef| `0.002381`
- `lag_00__CT5__alive`: coefficient `0.002333`, |coef| `0.002333`
- `lag_00__CT5__hp`: coefficient `0.002303`, |coef| `0.002303`

## Top 10 utility ridge features

- `lag_00__CT5__molly`: coefficient `0.002381` (raises CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.002105` (raises CT win probability)
- `lag_12__CT1__smoke`: coefficient `0.001943` (raises CT win probability)
- `lag_00__CT5__utility_total`: coefficient `0.001899` (raises CT win probability)
- `lag_11__CT_B_site_active_smokes`: coefficient `0.001532` (raises CT win probability)
- `lag_14__T_B_site_active_smokes`: coefficient `0.001414` (raises CT win probability)
- `lag_01__CT5__molly`: coefficient `0.001125` (raises CT win probability)
- `lag_11__CT_active_smokes`: coefficient `0.001086` (raises CT win probability)
- `lag_08__CT3__flash_duration`: coefficient `0.001037` (raises CT win probability)
- `lag_13__CT1__smoke`: coefficient `0.001035` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_SHOP`: coefficient `0.004195` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.003753` (raises CT win probability)
- `lag_06__T_duck_amount_mean`: coefficient `-0.003363` (lowers CT win probability)
- `lag_02__T4__is_scoped`: coefficient `-0.003299` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003082` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002730` (raises CT win probability)
- `lag_00__CT_place_SHOP`: coefficient `0.002496` (raises CT win probability)
- `lag_05__T_bomb_zone_count`: coefficient `-0.002485` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002464` (lowers CT win probability)
- `lag_03__CT_place_SHOP`: coefficient `0.002446` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `56223`, seconds `57.00`, LSTM delta `-0.3841`

Top all feature movements:
- `lag_02__CT_place_SHOP`: contribution `-0.021039`
- `lag_06__T_duck_amount_mean`: contribution `-0.019557`
- `lag_02__T4__is_scoped`: contribution `-0.015323`
- `lag_03__T_duck_amount_mean`: contribution `-0.014006`
- `lag_00__T_kills_last_3s`: contribution `-0.009765`

Top utility-only movements:
- `lag_00__CT5__molly`: contribution `-0.005906`

### tick `55359`, seconds `43.50`, LSTM delta `+0.2236`

Top all feature movements:
- `lag_05__T_bomb_zone_count`: contribution `+0.014464`
- `lag_06__T_duck_amount_mean`: contribution `+0.009778`
- `lag_00__kill_diff_last_3s`: contribution `+0.009034`
- `lag_06__T4__duck_amount`: contribution `+0.008823`
- `lag_14__T_bomb_zone_count`: contribution `+0.008753`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `54751`, seconds `34.00`, LSTM delta `-0.2200`

Top all feature movements:
- `lag_02__T4__is_scoped`: contribution `-0.015323`
- `lag_00__CT_shots_fired_sum`: contribution `-0.011347`
- `lag_15__CT_place_TRUCK`: contribution `-0.008286`
- `lag_05__CT_place_SHOP`: contribution `-0.006575`
- `lag_13__CT_place_SHOP`: contribution `-0.006242`

Top utility-only movements:
- `lag_12__CT3__flash_duration`: contribution `-0.004272`

### tick `54463`, seconds `29.50`, LSTM delta `+0.1499`

Top all feature movements:
- `lag_02__T4__is_scoped`: contribution `+0.015323`
- `lag_04__CT_place_SHOP`: contribution `+0.010839`
- `lag_00__kill_diff_last_3s`: contribution `+0.009034`
- `lag_08__CT3__flash_duration`: contribution `+0.007153`
- `lag_00__damage_diff_last_5s`: contribution `+0.006159`

Top utility-only movements:
- `lag_08__CT3__flash_duration`: contribution `+0.007153`

### tick `54111`, seconds `24.00`, LSTM delta `+0.1150`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009034`
- `lag_15__CT_place_TRUCK`: contribution `+0.008286`
- `lag_02__CT2__shots_fired`: contribution `+0.006210`
- `lag_00__CT_kills_last_3s`: contribution `+0.004887`
- `lag_02__CT_shots_fired_sum`: contribution `+0.004673`

Top utility-only movements:
- No utility movement among the top local contributors.
