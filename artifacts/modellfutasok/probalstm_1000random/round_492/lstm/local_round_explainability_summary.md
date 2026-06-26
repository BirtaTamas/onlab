# Local Round Explainability

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `17`

## Largest probability jumps

- tick `140378`, seconds `77.50`, LSTM `0.1375`, delta `-0.2246`
- tick `139002`, seconds `56.00`, LSTM `0.3128`, delta `-0.2234`
- tick `140666`, seconds `82.00`, LSTM `0.1662`, delta `-0.1249`
- tick `140858`, seconds `85.00`, LSTM `0.0351`, delta `-0.1087`
- tick `140346`, seconds `77.00`, LSTM `0.3621`, delta `+0.0964`
- tick `140506`, seconds `79.50`, LSTM `0.2079`, delta `+0.0742`
- tick `139034`, seconds `56.50`, LSTM `0.2470`, delta `-0.0658`
- tick `140570`, seconds `80.50`, LSTM `0.2907`, delta `+0.0527`
- tick `139354`, seconds `61.50`, LSTM `0.2972`, delta `+0.0481`
- tick `139482`, seconds `63.50`, LSTM `0.3404`, delta `+0.0462`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002943`, |coef| `0.002943`
- `lag_00__kill_diff_last_3s`: coefficient `0.002474`, |coef| `0.002474`
- `lag_00__T_damage_last_5s`: coefficient `-0.002396`, |coef| `0.002396`
- `lag_08__T_shots_fired_sum`: coefficient `0.002066`, |coef| `0.002066`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001594`, |coef| `0.001594`
- `lag_00__damage_diff_last_5s`: coefficient `0.001497`, |coef| `0.001497`
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001479`, |coef| `0.001479`
- `lag_15__CT2__duck_amount`: coefficient `-0.001460`, |coef| `0.001460`
- `lag_13__CT3__duck_amount`: coefficient `-0.001420`, |coef| `0.001420`
- `lag_00__CT3__alive`: coefficient `0.001385`, |coef| `0.001385`
- `lag_00__T2__is_walking`: coefficient `0.001383`, |coef| `0.001383`
- `lag_03__T_bomb_zone_count`: coefficient `-0.001378`, |coef| `0.001378`
- `lag_00__CT3__hp`: coefficient `0.001365`, |coef| `0.001365`
- `lag_01__T_kills_last_3s`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_00__CT3__armor`: coefficient `0.001309`, |coef| `0.001309`

## Top 10 utility ridge features

- `lag_06__T5__smoke`: coefficient `-0.001283` (lowers CT win probability)
- `lag_06__T2__smoke`: coefficient `0.001177` (raises CT win probability)
- `lag_05__T3__smoke`: coefficient `0.001171` (raises CT win probability)
- `lag_05__T1__smoke`: coefficient `-0.001134` (lowers CT win probability)
- `lag_04__T5__flash_duration`: coefficient `0.000901` (raises CT win probability)
- `lag_13__T5__flash_duration`: coefficient `-0.000834` (lowers CT win probability)
- `lag_03__CT_B_site_active_infernos`: coefficient `-0.000818` (lowers CT win probability)
- `lag_07__T2__smoke`: coefficient `0.000766` (raises CT win probability)
- `lag_06__T3__smoke`: coefficient `0.000766` (raises CT win probability)
- `lag_07__CT_B_site_active_infernos`: coefficient `0.000754` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.002943` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002474` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002396` (lowers CT win probability)
- `lag_08__T_shots_fired_sum`: coefficient `0.002066` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001594` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001497` (raises CT win probability)
- `lag_00__CT_place_UNKNOWN`: coefficient `0.001479` (raises CT win probability)
- `lag_15__CT2__duck_amount`: coefficient `-0.001460` (lowers CT win probability)
- `lag_13__CT3__duck_amount`: coefficient `-0.001420` (lowers CT win probability)
- `lag_00__CT3__alive`: coefficient `0.001385` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `140378`, seconds `77.50`, LSTM delta `-0.2246`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.015487`
- `lag_00__T_kills_last_3s`: contribution `-0.009323`
- `lag_03__T_bomb_zone_count`: contribution `-0.008022`
- `lag_00__T_shots_fired_sum`: contribution `-0.005974`
- `lag_00__kill_diff_last_3s`: contribution `-0.005954`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.004375`
- `lag_13__T5__flash_duration`: contribution `-0.004046`

### tick `139002`, seconds `56.00`, LSTM delta `-0.2234`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.009323`
- `lag_00__kill_diff_last_3s`: contribution `-0.005954`
- `lag_00__T_damage_last_5s`: contribution `-0.005744`
- `lag_15__CT2__duck_amount`: contribution `-0.005564`
- `lag_14__T1__is_scoped`: contribution `-0.005126`

Top utility-only movements:
- `lag_06__T5__smoke`: contribution `-0.002780`

### tick `140666`, seconds `82.00`, LSTM delta `-0.1249`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.010841`
- `lag_00__T_kills_last_3s`: contribution `-0.009323`
- `lag_00__T_shots_fired_sum`: contribution `-0.005974`
- `lag_00__T_damage_last_5s`: contribution `-0.005629`
- `lag_04__T1__is_scoped`: contribution `-0.005434`

Top utility-only movements:
- `lag_13__T5__flash_duration`: contribution `+0.004046`

### tick `140858`, seconds `85.00`, LSTM delta `-0.1087`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.005954`
- `lag_00__T_damage_last_5s`: contribution `-0.005400`
- `lag_00__T_shots_fired_sum`: contribution `-0.004779`
- `lag_15__T1__is_scoped`: contribution `+0.004269`
- `lag_08__T_bomb_zone_count`: contribution `-0.004060`

Top utility-only movements:
- `lag_07__CT_B_site_active_infernos`: contribution `-0.002592`

### tick `140346`, seconds `77.00`, LSTM delta `+0.0964`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `+0.012389`
- `lag_07__T_shots_fired_sum`: contribution `+0.003902`
- `lag_02__T_bomb_zone_count`: contribution `+0.003875`
- `lag_07__CT_place_HOUSE`: contribution `+0.003655`
- `lag_07__T_place_TSIDELOWER`: contribution `+0.003436`

Top utility-only movements:
- `lag_03__T5__flash_duration`: contribution `+0.001576`
