# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `11`

## Largest probability jumps

- tick `107735`, seconds `84.50`, LSTM `0.9287`, delta `+0.1913`
- tick `103959`, seconds `25.50`, LSTM `0.6577`, delta `-0.1477`
- tick `106775`, seconds `69.50`, LSTM `0.8711`, delta `+0.1457`
- tick `105367`, seconds `47.50`, LSTM `0.6697`, delta `-0.1349`
- tick `103607`, seconds `20.00`, LSTM `0.8255`, delta `+0.1305`
- tick `105111`, seconds `43.50`, LSTM `0.7308`, delta `+0.1153`
- tick `103255`, seconds `14.50`, LSTM `0.6733`, delta `+0.0616`
- tick `104439`, seconds `33.00`, LSTM `0.6647`, delta `+0.0527`
- tick `105207`, seconds `45.00`, LSTM `0.7967`, delta `+0.0383`
- tick `103799`, seconds `23.00`, LSTM `0.8191`, delta `-0.0358`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004255`, |coef| `0.004255`
- `lag_00__CT_kills_last_3s`: coefficient `0.004088`, |coef| `0.004088`
- `lag_00__T_bomb_zone_count`: coefficient `-0.003431`, |coef| `0.003431`
- `lag_00__damage_diff_last_5s`: coefficient `0.003408`, |coef| `0.003408`
- `lag_10__T_bomb_zone_count`: coefficient `0.003391`, |coef| `0.003391`
- `lag_00__CT_damage_last_5s`: coefficient `0.002789`, |coef| `0.002789`
- `lag_01__T1__duck_amount`: coefficient `-0.002009`, |coef| `0.002009`
- `lag_05__T5__duck_amount`: coefficient `0.001860`, |coef| `0.001860`
- `lag_00__T_macro_B`: coefficient `-0.001719`, |coef| `0.001719`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001719`, |coef| `0.001719`
- `lag_04__CT5__is_walking`: coefficient `0.001663`, |coef| `0.001663`
- `lag_09__T_B_site_active_infernos`: coefficient `0.001653`, |coef| `0.001653`
- `lag_13__CT5__is_walking`: coefficient `0.001546`, |coef| `0.001546`
- `lag_00__CT3__is_scoped`: coefficient `-0.001537`, |coef| `0.001537`
- `lag_05__T3__is_walking`: coefficient `0.001509`, |coef| `0.001509`

## Top 10 utility ridge features

- `lag_09__T_B_site_active_infernos`: coefficient `0.001653` (raises CT win probability)
- `lag_04__T5__flash_duration`: coefficient `-0.001405` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `0.001265` (raises CT win probability)
- `lag_12__CT3__flash_duration`: coefficient `0.001261` (raises CT win probability)
- `lag_09__T_active_infernos`: coefficient `0.001207` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.001122` (lowers CT win probability)
- `lag_12__CT_utility_damage_last_5s`: coefficient `0.001116` (raises CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `0.001016` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000949` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `-0.000938` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.004255` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004088` (raises CT win probability)
- `lag_00__T_bomb_zone_count`: coefficient `-0.003431` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003408` (raises CT win probability)
- `lag_10__T_bomb_zone_count`: coefficient `0.003391` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002789` (raises CT win probability)
- `lag_01__T1__duck_amount`: coefficient `-0.002009` (lowers CT win probability)
- `lag_05__T5__duck_amount`: coefficient `0.001860` (raises CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.001719` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.001719` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `107735`, seconds `84.50`, LSTM delta `+0.1913`

Top all feature movements:
- `lag_00__T_bomb_zone_count`: contribution `+0.019972`
- `lag_10__T_bomb_zone_count`: contribution `+0.019742`
- `lag_00__CT_kills_last_3s`: contribution `+0.011802`
- `lag_00__kill_diff_last_3s`: contribution `+0.010243`
- `lag_01__T1__duck_amount`: contribution `+0.007866`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `103959`, seconds `25.50`, LSTM delta `-0.1477`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.010243`
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.008602`
- `lag_00__damage_diff_last_5s`: contribution `-0.007611`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.005758`
- `lag_14__CT_flashed_players`: contribution `-0.005468`

Top utility-only movements:
- `lag_12__CT_utility_damage_last_5s`: contribution `-0.008602`
- `lag_12__utility_damage_diff_last_5s`: contribution `-0.005758`
- `lag_04__T3__flash_duration`: contribution `-0.004238`
- `lag_05__CT2__flash_duration`: contribution `-0.003644`
- `lag_14__CT2__flash_duration`: contribution `-0.003230`

### tick `106775`, seconds `69.50`, LSTM delta `+0.1457`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011802`
- `lag_00__kill_diff_last_3s`: contribution `+0.010243`
- `lag_00__CT3__is_scoped`: contribution `+0.006990`
- `lag_05__T5__duck_amount`: contribution `+0.006796`
- `lag_00__CT_damage_last_5s`: contribution `+0.006080`

Top utility-only movements:
- `lag_09__T_B_site_active_infernos`: contribution `+0.004673`

### tick `105367`, seconds `47.50`, LSTM delta `-0.1349`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.010243`
- `lag_04__T5__flash_duration`: contribution `-0.009930`
- `lag_02__CT1__flash_duration`: contribution `-0.006982`
- `lag_01__T1__duck_amount`: contribution `-0.005784`
- `lag_00__damage_diff_last_5s`: contribution `-0.005612`

Top utility-only movements:
- `lag_04__T5__flash_duration`: contribution `-0.009930`
- `lag_02__CT1__flash_duration`: contribution `-0.006982`
- `lag_04__T_flash_duration_sum`: contribution `-0.002726`
- `lag_02__CT_flash_duration_sum`: contribution `-0.001870`

### tick `103607`, seconds `20.00`, LSTM delta `+0.1305`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.011802`
- `lag_00__kill_diff_last_3s`: contribution `+0.010243`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.009745`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.007230`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.006419`

Top utility-only movements:
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.009745`
- `lag_01__CT_utility_damage_last_5s`: contribution `+0.007230`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.006419`
- `lag_08__T2__flash_duration`: contribution `+0.005386`
- `lag_01__utility_damage_diff_last_5s`: contribution `+0.004704`
