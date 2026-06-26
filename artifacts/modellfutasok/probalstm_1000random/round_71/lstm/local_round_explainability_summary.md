# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `8`

## Largest probability jumps

- tick `68714`, seconds `79.00`, LSTM `0.7899`, delta `+0.2647`
- tick `66570`, seconds `45.50`, LSTM `0.4083`, delta `+0.2433`
- tick `66122`, seconds `38.50`, LSTM `0.1438`, delta `-0.2419`
- tick `67786`, seconds `64.50`, LSTM `0.5865`, delta `+0.1844`
- tick `66922`, seconds `51.00`, LSTM `0.4736`, delta `+0.1023`
- tick `69002`, seconds `83.50`, LSTM `0.9457`, delta `+0.1003`
- tick `65578`, seconds `30.00`, LSTM `0.4857`, delta `-0.0795`
- tick `66890`, seconds `50.50`, LSTM `0.3713`, delta `-0.0625`
- tick `67658`, seconds `62.50`, LSTM `0.4147`, delta `+0.0598`
- tick `67978`, seconds `67.50`, LSTM `0.5609`, delta `-0.0593`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005059`, |coef| `0.005059`
- `lag_00__CT_kills_last_3s`: coefficient `0.004802`, |coef| `0.004802`
- `lag_11__T_place_ALLEY`: coefficient `-0.004123`, |coef| `0.004123`
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002969`, |coef| `0.002969`
- `lag_00__T_macro_B`: coefficient `-0.002969`, |coef| `0.002969`
- `lag_04__CT4__is_scoped`: coefficient `-0.002892`, |coef| `0.002892`
- `lag_00__CT_damage_last_5s`: coefficient `0.002823`, |coef| `0.002823`
- `lag_00__T5__alive`: coefficient `-0.002782`, |coef| `0.002782`
- `lag_00__T5__hp`: coefficient `-0.002733`, |coef| `0.002733`
- `lag_00__CT2__duck_amount`: coefficient `0.002662`, |coef| `0.002662`
- `lag_00__T5__shots_fired`: coefficient `0.002632`, |coef| `0.002632`
- `lag_00__T5__armor`: coefficient `-0.002631`, |coef| `0.002631`
- `lag_06__T5__is_walking`: coefficient `-0.002535`, |coef| `0.002535`
- `lag_00__damage_diff_last_5s`: coefficient `0.002388`, |coef| `0.002388`
- `lag_00__T5__has_helmet`: coefficient `-0.002382`, |coef| `0.002382`

## Top 10 utility ridge features

- `lag_07__CT2__flash_duration`: coefficient `-0.001818` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001490` (lowers CT win probability)
- `lag_03__T_B_site_active_smokes`: coefficient `-0.001487` (lowers CT win probability)
- `lag_07__T4__flash_duration`: coefficient `-0.001210` (lowers CT win probability)
- `lag_10__T3__flash_duration`: coefficient `0.001192` (raises CT win probability)
- `lag_08__CT1__flash_duration`: coefficient `0.001150` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `-0.001126` (lowers CT win probability)
- `lag_05__CT4__flash_duration`: coefficient `0.001076` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.001059` (lowers CT win probability)
- `lag_03__T_active_smokes`: coefficient `-0.001039` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005059` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004802` (raises CT win probability)
- `lag_11__T_place_ALLEY`: coefficient `-0.004123` (lowers CT win probability)
- `lag_00__T_place_BOMBSITEB`: coefficient `-0.002969` (lowers CT win probability)
- `lag_00__T_macro_B`: coefficient `-0.002969` (lowers CT win probability)
- `lag_04__CT4__is_scoped`: coefficient `-0.002892` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002823` (raises CT win probability)
- `lag_00__T5__alive`: coefficient `-0.002782` (lowers CT win probability)
- `lag_00__T5__hp`: coefficient `-0.002733` (lowers CT win probability)
- `lag_00__CT2__duck_amount`: coefficient `0.002662` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `68714`, seconds `79.00`, LSTM delta `+0.2647`

Top all feature movements:
- `lag_11__T_place_ALLEY`: contribution `+0.017469`
- `lag_00__CT_kills_last_3s`: contribution `+0.013864`
- `lag_00__kill_diff_last_3s`: contribution `+0.012178`
- `lag_00__CT2__duck_amount`: contribution `+0.010141`
- `lag_04__CT4__is_scoped`: contribution `+0.009858`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66570`, seconds `45.50`, LSTM delta `+0.2433`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013864`
- `lag_00__kill_diff_last_3s`: contribution `+0.012178`
- `lag_09__T3__is_scoped`: contribution `+0.008860`
- `lag_00__T3__flash_duration`: contribution `+0.008343`
- `lag_07__T3__is_scoped`: contribution `+0.006160`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.008343`
- `lag_08__CT1__flash_duration`: contribution `+0.005801`
- `lag_01__T4__flash_duration`: contribution `+0.003507`

### tick `66122`, seconds `38.50`, LSTM delta `-0.2419`

Top all feature movements:
- `lag_02__T_shots_fired_sum`: contribution `-0.019138`
- `lag_02__T5__shots_fired`: contribution `-0.018935`
- `lag_00__kill_diff_last_3s`: contribution `-0.012178`
- `lag_07__CT2__flash_duration`: contribution `-0.010161`
- `lag_07__T4__flash_duration`: contribution `-0.007336`

Top utility-only movements:
- `lag_07__CT2__flash_duration`: contribution `-0.010161`
- `lag_07__T4__flash_duration`: contribution `-0.007336`
- `lag_07__T5__flash_duration`: contribution `-0.004987`
- `lag_07__T_flash_duration_sum`: contribution `-0.004921`

### tick `67786`, seconds `64.50`, LSTM delta `+0.1844`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013864`
- `lag_00__kill_diff_last_3s`: contribution `+0.012178`
- `lag_05__T_bomb_zone_count`: contribution `+0.010523`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008387`
- `lag_00__CT2__duck_amount`: contribution `+0.008340`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `66922`, seconds `51.00`, LSTM delta `+0.1023`

Top all feature movements:
- `lag_00__CT_kills_last_3s`: contribution `+0.013864`
- `lag_00__kill_diff_last_3s`: contribution `+0.012178`
- `lag_15__T4__duck_amount`: contribution `+0.005640`
- `lag_00__T1__flash_duration`: contribution `+0.005183`
- `lag_07__CT2__flash_duration`: contribution `+0.004729`

Top utility-only movements:
- `lag_00__T1__flash_duration`: contribution `+0.005183`
- `lag_07__CT2__flash_duration`: contribution `+0.004729`
- `lag_11__T_B_site_active_infernos`: contribution `+0.002642`
