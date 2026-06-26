# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `5`

## Largest probability jumps

- tick `40425`, seconds `93.50`, LSTM `0.8001`, delta `+0.2577`
- tick `36617`, seconds `34.00`, LSTM `0.6218`, delta `-0.1952`
- tick `36297`, seconds `29.00`, LSTM `0.5854`, delta `+0.1575`
- tick `35465`, seconds `16.00`, LSTM `0.3694`, delta `-0.1554`
- tick `40233`, seconds `90.50`, LSTM `0.5169`, delta `+0.1347`
- tick `36489`, seconds `32.00`, LSTM `0.7500`, delta `+0.1218`
- tick `40521`, seconds `95.00`, LSTM `0.7261`, delta `-0.0972`
- tick `39817`, seconds `84.00`, LSTM `0.5251`, delta `+0.0781`
- tick `40105`, seconds `88.50`, LSTM `0.4554`, delta `-0.0546`
- tick `39529`, seconds `79.50`, LSTM `0.4095`, delta `-0.0508`

## Top 15 local ridge features

- `lag_06__CT_defusing_count`: coefficient `0.004404`, |coef| `0.004404`
- `lag_00__kill_diff_last_3s`: coefficient `0.002688`, |coef| `0.002688`
- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.002030`, |coef| `0.002030`
- `lag_00__CT_kills_last_3s`: coefficient `0.002028`, |coef| `0.002028`
- `lag_00__damage_diff_last_5s`: coefficient `0.001831`, |coef| `0.001831`
- `lag_05__CT_defusing_count`: coefficient `0.001804`, |coef| `0.001804`
- `lag_00__CT_defusing_count`: coefficient `0.001788`, |coef| `0.001788`
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.001665`, |coef| `0.001665`
- `lag_00__T_duck_amount_mean`: coefficient `-0.001637`, |coef| `0.001637`
- `lag_07__CT_defusing_count`: coefficient `0.001614`, |coef| `0.001614`
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001556`, |coef| `0.001556`
- `lag_00__CT_damage_last_5s`: coefficient `0.001501`, |coef| `0.001501`
- `lag_00__T5__is_walking`: coefficient `-0.001482`, |coef| `0.001482`
- `lag_08__T_place_TSTAIRS`: coefficient `0.001427`, |coef| `0.001427`
- `lag_13__CT_place_CONNECTOR`: coefficient `-0.001411`, |coef| `0.001411`

## Top 10 utility ridge features

- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.002030` (lowers CT win probability)
- `lag_10__utility_damage_diff_last_5s`: coefficient `-0.001665` (lowers CT win probability)
- `lag_08__CT_B_site_active_infernos`: coefficient `-0.001556` (lowers CT win probability)
- `lag_15__CT3__smoke`: coefficient `-0.001075` (lowers CT win probability)
- `lag_07__CT3__flash`: coefficient `-0.000995` (lowers CT win probability)
- `lag_08__CT_active_infernos`: coefficient `-0.000949` (lowers CT win probability)
- `lag_04__T_B_site_active_smokes`: coefficient `-0.000899` (lowers CT win probability)
- `lag_04__CT_utility_damage_last_5s`: coefficient `-0.000816` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.000780` (lowers CT win probability)
- `lag_09__CT_utility_damage_last_5s`: coefficient `-0.000753` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_06__CT_defusing_count`: coefficient `0.004404` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002688` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002028` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001831` (raises CT win probability)
- `lag_05__CT_defusing_count`: coefficient `0.001804` (raises CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.001788` (raises CT win probability)
- `lag_00__T_duck_amount_mean`: coefficient `-0.001637` (lowers CT win probability)
- `lag_07__CT_defusing_count`: coefficient `0.001614` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001501` (raises CT win probability)
- `lag_00__T5__is_walking`: coefficient `-0.001482` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `40425`, seconds `93.50`, LSTM delta `+0.2577`

Top all feature movements:
- `lag_06__CT_defusing_count`: contribution `+0.042696`
- `lag_10__CT_utility_damage_last_5s`: contribution `+0.008267`
- `lag_05__T5__is_scoped`: contribution `+0.006706`
- `lag_00__kill_diff_last_3s`: contribution `+0.006470`
- `lag_00__CT_kills_last_3s`: contribution `+0.005855`

Top utility-only movements:
- `lag_10__CT_utility_damage_last_5s`: contribution `+0.008267`
- `lag_10__utility_damage_diff_last_5s`: contribution `+0.005561`
- `lag_08__CT_B_site_active_infernos`: contribution `+0.005344`

### tick `36617`, seconds `34.00`, LSTM delta `-0.1952`

Top all feature movements:
- `lag_08__T_place_TSTAIRS`: contribution `-0.008090`
- `lag_00__CT_place_CANAL`: contribution `-0.007004`
- `lag_00__kill_diff_last_3s`: contribution `-0.006470`
- `lag_00__damage_diff_last_5s`: contribution `-0.006031`
- `lag_02__T5__is_scoped`: contribution `-0.005252`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `36297`, seconds `29.00`, LSTM delta `+0.1575`

Top all feature movements:
- `lag_13__CT_place_CANAL`: contribution `+0.008168`
- `lag_08__T_place_TSTAIRS`: contribution `+0.008090`
- `lag_12__T_place_STREET`: contribution `+0.007575`
- `lag_00__kill_diff_last_3s`: contribution `+0.006470`
- `lag_00__CT_kills_last_3s`: contribution `+0.005855`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `35465`, seconds `16.00`, LSTM delta `-0.1554`

Top all feature movements:
- `lag_12__T_place_STREET`: contribution `-0.007575`
- `lag_00__CT_place_MAIN`: contribution `-0.007086`
- `lag_00__kill_diff_last_3s`: contribution `-0.006470`
- `lag_09__CT_place_MAIN`: contribution `-0.006173`
- `lag_13__CT_place_CONNECTOR`: contribution `-0.005046`

Top utility-only movements:
- `lag_00__CT_A_site_active_infernos`: contribution `-0.002054`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.002041`

### tick `40233`, seconds `90.50`, LSTM delta `+0.1347`

Top all feature movements:
- `lag_00__CT_defusing_count`: contribution `+0.017332`
- `lag_15__T_bomb_zone_count`: contribution `+0.004993`
- `lag_02__T2__duck_amount`: contribution `+0.004675`
- `lag_15__CT_place_SNIPERSNEST`: contribution `+0.004436`
- `lag_13__T_place_CONNECTOR`: contribution `+0.004320`

Top utility-only movements:
- `lag_04__CT_utility_damage_last_5s`: contribution `+0.003322`
- `lag_14__CT_utility_damage_last_5s`: contribution `+0.003055`
- `lag_02__CT_B_site_active_infernos`: contribution `+0.002363`
- `lag_04__utility_damage_diff_last_5s`: contribution `+0.002229`
- `lag_13__CT_B_site_active_infernos`: contribution `+0.002171`
