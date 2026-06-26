# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `8`

## Largest probability jumps

- tick `72894`, seconds `12.50`, LSTM `0.0590`, delta `-0.1575`
- tick `72126`, seconds `0.50`, LSTM `0.2197`, delta `-0.0554`
- tick `76830`, seconds `74.00`, LSTM `0.0976`, delta `+0.0531`
- tick `77214`, seconds `80.00`, LSTM `0.0899`, delta `-0.0437`
- tick `77470`, seconds `84.00`, LSTM `0.0533`, delta `-0.0349`
- tick `77246`, seconds `80.50`, LSTM `0.1233`, delta `+0.0334`
- tick `77086`, seconds `78.00`, LSTM `0.1309`, delta `+0.0251`
- tick `77278`, seconds `81.00`, LSTM `0.1023`, delta `-0.0210`
- tick `72638`, seconds `8.50`, LSTM `0.1731`, delta `+0.0192`
- tick `76702`, seconds `72.00`, LSTM `0.0295`, delta `-0.0180`

## Top 15 local ridge features

- `lag_15__T_place_LOWERMID`: coefficient `-0.001309`, |coef| `0.001309`
- `lag_00__kill_diff_last_3s`: coefficient `0.000888`, |coef| `0.000888`
- `lag_02__CT_place_APARTMENTS`: coefficient `-0.000875`, |coef| `0.000875`
- `lag_00__T_bomb_zone_count`: coefficient `-0.000736`, |coef| `0.000736`
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.000726`, |coef| `0.000726`
- `lag_01__T_money_sum`: coefficient `-0.000709`, |coef| `0.000709`
- `lag_01__T_start_balance_sum`: coefficient `-0.000697`, |coef| `0.000697`
- `lag_11__T_place_LOWERMID`: coefficient `0.000686`, |coef| `0.000686`
- `lag_12__CT_place_LIBRARY`: coefficient `-0.000651`, |coef| `0.000651`
- `lag_06__T_bomb_zone_count`: coefficient `-0.000650`, |coef| `0.000650`
- `lag_01__T_place_SECONDMID`: coefficient `0.000645`, |coef| `0.000645`
- `lag_02__T_shots_fired_sum`: coefficient `-0.000633`, |coef| `0.000633`
- `lag_00__T_kills_last_3s`: coefficient `-0.000625`, |coef| `0.000625`
- `lag_01__T3__shots_fired`: coefficient `0.000604`, |coef| `0.000604`
- `lag_14__CT_place_LIBRARY`: coefficient `0.000602`, |coef| `0.000602`

## Top 10 utility ridge features

- `lag_03__T5__flash_duration`: coefficient `-0.000591` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.000585` (lowers CT win probability)
- `lag_02__CT3__flash_duration`: coefficient `-0.000569` (lowers CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `-0.000497` (lowers CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `0.000399` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000374` (lowers CT win probability)
- `lag_15__T2__flash_duration`: coefficient `0.000334` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000334` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.000303` (lowers CT win probability)
- `lag_06__T5__molly`: coefficient `0.000298` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_15__T_place_LOWERMID`: coefficient `-0.001309` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000888` (raises CT win probability)
- `lag_02__CT_place_APARTMENTS`: coefficient `-0.000875` (lowers CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.000736` (lowers CT win probability)
- `lag_00__CT_place_APARTMENTS`: coefficient `-0.000726` (lowers CT win probability)
- `lag_01__T_money_sum`: coefficient `-0.000709` (lowers CT win probability)
- `lag_01__T_start_balance_sum`: coefficient `-0.000697` (lowers CT win probability)
- `lag_11__T_place_LOWERMID`: coefficient `0.000686` (raises CT win probability)
- `lag_12__CT_place_LIBRARY`: coefficient `-0.000651` (lowers CT win probability)
- `lag_06__T_bomb_zone_count`: coefficient `-0.000650` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `72894`, seconds `12.50`, LSTM delta `-0.1575`

Top all feature movements:
- `lag_15__T_place_LOWERMID`: contribution `-0.013063`
- `lag_11__T_place_LOWERMID`: contribution `-0.004560`
- `lag_01__T_place_SECONDMID`: contribution `-0.004221`
- `lag_12__CT_place_LIBRARY`: contribution `-0.004174`
- `lag_14__CT_place_LIBRARY`: contribution `-0.003861`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `-0.002946`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.002771`
- `lag_01__utility_damage_diff_last_5s`: contribution `-0.001929`

### tick `72126`, seconds `0.50`, LSTM delta `-0.0554`

Top all feature movements:
- `lag_01__T_money_sum`: contribution `-0.003333`
- `lag_01__T_start_balance_sum`: contribution `-0.003275`
- `lag_01__CT_place_CTSPAWN`: contribution `-0.002051`
- `lag_00__T_velocity_mean`: contribution `-0.001777`
- `lag_01__T_place_TSPAWN`: contribution `-0.001677`

Top utility-only movements:
- `lag_01__molly_inv_diff`: contribution `-0.000663`
- `lag_01__T_smoke_inv`: contribution `-0.000341`
- `lag_01__T_molly_inv`: contribution `-0.000336`

### tick `76830`, seconds `74.00`, LSTM delta `+0.0531`

Top all feature movements:
- `lag_02__CT3__flash_duration`: contribution `+0.004001`
- `lag_07__T_shots_fired_sum`: contribution `+0.002809`
- `lag_05__T1__duck_amount`: contribution `+0.002199`
- `lag_00__kill_diff_last_3s`: contribution `+0.002136`
- `lag_02__T_shots_fired_sum`: contribution `+0.001898`

Top utility-only movements:
- `lag_02__CT3__flash_duration`: contribution `+0.004001`
- `lag_15__T2__flash_duration`: contribution `+0.001046`
- `lag_02__CT_flash_duration_sum`: contribution `+0.000752`

### tick `77214`, seconds `80.00`, LSTM delta `-0.0437`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `-0.004285`
- `lag_14__CT3__flash_duration`: contribution `-0.002809`
- `lag_10__CT_place_UNDERPASS`: contribution `-0.002525`
- `lag_10__CT4__duck_amount`: contribution `-0.001736`
- `lag_14__T2__duck_amount`: contribution `-0.001701`

Top utility-only movements:
- `lag_14__CT3__flash_duration`: contribution `-0.002809`
- `lag_14__CT_flash_duration_sum`: contribution `-0.000606`

### tick `77470`, seconds `84.00`, LSTM delta `-0.0349`

Top all feature movements:
- `lag_06__T_bomb_zone_count`: contribution `-0.003786`
- `lag_08__T_bomb_zone_count`: contribution `-0.002216`
- `lag_05__T1__duck_amount`: contribution `-0.002163`
- `lag_00__kill_diff_last_3s`: contribution `-0.002136`
- `lag_00__T_kills_last_3s`: contribution `-0.001981`

Top utility-only movements:
- No utility movement among the top local contributors.
