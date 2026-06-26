# Local Round Explainability

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `3`

## Largest probability jumps

- tick `18702`, seconds `46.00`, LSTM `0.5099`, delta `+0.3281`
- tick `18510`, seconds `43.00`, LSTM `0.2847`, delta `-0.2906`
- tick `17550`, seconds `28.00`, LSTM `0.8449`, delta `+0.2865`
- tick `16526`, seconds `12.00`, LSTM `0.5506`, delta `-0.2301`
- tick `17646`, seconds `29.50`, LSTM `0.6092`, delta `-0.2275`
- tick `18158`, seconds `37.50`, LSTM `0.7335`, delta `+0.2261`
- tick `19022`, seconds `51.00`, LSTM `0.7262`, delta `+0.1758`
- tick `17006`, seconds `19.50`, LSTM `0.5327`, delta `-0.1603`
- tick `16686`, seconds `14.50`, LSTM `0.4693`, delta `+0.1578`
- tick `17486`, seconds `27.00`, LSTM `0.6043`, delta `+0.1219`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006948`, |coef| `0.006948`
- `lag_15__CT_place_UNDERPASS`: coefficient `-0.005444`, |coef| `0.005444`
- `lag_00__T_kills_last_3s`: coefficient `-0.004733`, |coef| `0.004733`
- `lag_00__CT_defusing_count`: coefficient `0.004682`, |coef| `0.004682`
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.004141`, |coef| `0.004141`
- `lag_00__T_flash_alpha_mean`: coefficient `-0.004047`, |coef| `0.004047`
- `lag_00__CT_kills_last_3s`: coefficient `0.004021`, |coef| `0.004021`
- `lag_00__damage_diff_last_5s`: coefficient `0.003693`, |coef| `0.003693`
- `lag_00__CT_velocity_mean`: coefficient `-0.003507`, |coef| `0.003507`
- `lag_03__CT_place_UNDERPASS`: coefficient `0.003432`, |coef| `0.003432`
- `lag_04__T_place_TRUCK`: coefficient `0.003323`, |coef| `0.003323`
- `lag_00__T_place_TRUCK`: coefficient `-0.003092`, |coef| `0.003092`
- `lag_14__T3__has_bomb`: coefficient `0.003044`, |coef| `0.003044`
- `lag_05__T_place_TRUCK`: coefficient `0.003036`, |coef| `0.003036`
- `lag_10__T_flash_alpha_mean`: coefficient `-0.002956`, |coef| `0.002956`

## Top 10 utility ridge features

- `lag_00__T_flash_alpha_mean`: coefficient `-0.004047` (lowers CT win probability)
- `lag_10__T_flash_alpha_mean`: coefficient `-0.002956` (lowers CT win probability)
- `lag_00__CT_utility_damage_last_5s`: coefficient `0.002412` (raises CT win probability)
- `lag_15__CT_utility_damage_last_5s`: coefficient `-0.002328` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001979` (raises CT win probability)
- `lag_15__utility_damage_diff_last_5s`: coefficient `-0.001918` (lowers CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.001803` (lowers CT win probability)
- `lag_11__T_flash_alpha_mean`: coefficient `-0.001727` (lowers CT win probability)
- `lag_10__CT_utility_damage_last_5s`: coefficient `-0.001547` (lowers CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.001490` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.006948` (raises CT win probability)
- `lag_15__CT_place_UNDERPASS`: coefficient `-0.005444` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.004733` (lowers CT win probability)
- `lag_00__CT_defusing_count`: coefficient `0.004682` (raises CT win probability)
- `lag_14__CT_place_UNDERPASS`: coefficient `-0.004141` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004021` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003693` (raises CT win probability)
- `lag_00__CT_velocity_mean`: coefficient `-0.003507` (lowers CT win probability)
- `lag_03__CT_place_UNDERPASS`: coefficient `0.003432` (raises CT win probability)
- `lag_04__T_place_TRUCK`: coefficient `0.003323` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `18702`, seconds `46.00`, LSTM delta `+0.3281`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.033448`
- `lag_15__CT_place_UNDERPASS`: contribution `+0.031569`
- `lag_00__T_flash_alpha_mean`: contribution `+0.024555`
- `lag_00__T_kills_last_3s`: contribution `+0.014996`
- `lag_02__T_duck_amount_mean`: contribution `+0.014905`

Top utility-only movements:
- `lag_00__T_flash_alpha_mean`: contribution `+0.024555`
- `lag_11__CT_utility_damage_last_5s`: contribution `+0.008337`
- `lag_11__utility_damage_diff_last_5s`: contribution `+0.005652`

### tick `18510`, seconds `43.00`, LSTM delta `-0.2906`

Top all feature movements:
- `lag_14__CT_place_UNDERPASS`: contribution `-0.024010`
- `lag_14__T_bomb_zone_count`: contribution `-0.016827`
- `lag_00__kill_diff_last_3s`: contribution `-0.016724`
- `lag_09__CT_place_UNDERPASS`: contribution `-0.016670`
- `lag_00__T_kills_last_3s`: contribution `-0.014996`

Top utility-only movements:
- `lag_15__CT_utility_damage_last_5s`: contribution `-0.010760`
- `lag_15__utility_damage_diff_last_5s`: contribution `-0.007272`

### tick `17550`, seconds `28.00`, LSTM delta `+0.2865`

Top all feature movements:
- `lag_04__T_place_TRUCK`: contribution `+0.057708`
- `lag_02__T_place_TRUCK`: contribution `+0.047087`
- `lag_00__kill_diff_last_3s`: contribution `+0.016724`
- `lag_00__CT_kills_last_3s`: contribution `+0.011609`
- `lag_00__damage_diff_last_5s`: contribution `+0.008332`

Top utility-only movements:
- `lag_14__CT_A_site_active_infernos`: contribution `+0.004290`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.003805`

### tick `16526`, seconds `12.00`, LSTM delta `-0.2301`

Top all feature movements:
- `lag_13__CT_place_SHOP`: contribution `-0.026346`
- `lag_01__CT_place_TRUCK`: contribution `-0.014318`
- `lag_13__T_place_HOUSE`: contribution `-0.010729`
- `lag_09__CT_place_SHOP`: contribution `-0.010703`
- `lag_00__damage_diff_last_5s`: contribution `-0.008332`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `-0.004357`

### tick `17646`, seconds `29.50`, LSTM delta `-0.2275`

Top all feature movements:
- `lag_05__T_place_TRUCK`: contribution `-0.052722`
- `lag_07__T_place_TRUCK`: contribution `-0.040132`
- `lag_00__kill_diff_last_3s`: contribution `-0.016724`
- `lag_00__T_kills_last_3s`: contribution `-0.014996`
- `lag_06__CT_shots_fired_sum`: contribution `-0.007015`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `-0.002961`
