# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `115190`, seconds `20.00`, LSTM `0.5542`, delta `+0.2385`
- tick `116278`, seconds `37.00`, LSTM `0.2296`, delta `-0.2140`
- tick `116182`, seconds `35.50`, LSTM `0.4665`, delta `+0.2006`
- tick `115318`, seconds `22.00`, LSTM `0.4659`, delta `-0.1464`
- tick `115382`, seconds `23.00`, LSTM `0.3294`, delta `-0.0879`
- tick `115766`, seconds `29.00`, LSTM `0.3152`, delta `-0.0661`
- tick `115958`, seconds `32.00`, LSTM `0.2504`, delta `+0.0652`
- tick `114966`, seconds `16.50`, LSTM `0.4038`, delta `-0.0630`
- tick `116150`, seconds `35.00`, LSTM `0.2659`, delta `+0.0506`
- tick `115350`, seconds `22.50`, LSTM `0.4173`, delta `-0.0486`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002493`, |coef| `0.002493`
- `lag_00__damage_diff_last_5s`: coefficient `0.002427`, |coef| `0.002427`
- `lag_02__T_place_TRUCK`: coefficient `-0.002377`, |coef| `0.002377`
- `lag_05__CT5__duck_amount`: coefficient `0.002119`, |coef| `0.002119`
- `lag_00__CT_kills_last_3s`: coefficient `0.002117`, |coef| `0.002117`
- `lag_09__T4__duck_amount`: coefficient `-0.002069`, |coef| `0.002069`
- `lag_00__T_place_TRUCK`: coefficient `-0.002040`, |coef| `0.002040`
- `lag_08__CT_place_STAIRS`: coefficient `0.002024`, |coef| `0.002024`
- `lag_00__T_place_CATWALK`: coefficient `-0.001962`, |coef| `0.001962`
- `lag_12__CT_place_PALACEINTERIOR`: coefficient `0.001935`, |coef| `0.001935`
- `lag_01__T_place_TRUCK`: coefficient `-0.001930`, |coef| `0.001930`
- `lag_07__CT_place_STAIRS`: coefficient `-0.001892`, |coef| `0.001892`
- `lag_09__T_place_TRUCK`: coefficient `-0.001871`, |coef| `0.001871`
- `lag_13__T_place_TRUCK`: coefficient `0.001869`, |coef| `0.001869`
- `lag_00__CT_damage_last_5s`: coefficient `0.001755`, |coef| `0.001755`

## Top 10 utility ridge features

- `lag_07__CT3__flash_duration`: coefficient `0.001610` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.001564` (raises CT win probability)
- `lag_00__T4__flash_duration`: coefficient `-0.001395` (lowers CT win probability)
- `lag_10__T1__flash_duration`: coefficient `-0.001148` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.001144` (raises CT win probability)
- `lag_07__T4__flash_duration`: coefficient `0.001092` (raises CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001085` (raises CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.001031` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.000836` (raises CT win probability)
- `lag_13__CT3__flash_duration`: coefficient `-0.000791` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002493` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.002427` (raises CT win probability)
- `lag_02__T_place_TRUCK`: coefficient `-0.002377` (lowers CT win probability)
- `lag_05__CT5__duck_amount`: coefficient `0.002119` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002117` (raises CT win probability)
- `lag_09__T4__duck_amount`: coefficient `-0.002069` (lowers CT win probability)
- `lag_00__T_place_TRUCK`: coefficient `-0.002040` (lowers CT win probability)
- `lag_08__CT_place_STAIRS`: coefficient `0.002024` (raises CT win probability)
- `lag_00__T_place_CATWALK`: coefficient `-0.001962` (lowers CT win probability)
- `lag_12__CT_place_PALACEINTERIOR`: coefficient `0.001935` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `115190`, seconds `20.00`, LSTM delta `+0.2385`

Top all feature movements:
- `lag_08__CT_place_STAIRS`: contribution `+0.015754`
- `lag_07__CT_place_STAIRS`: contribution `+0.014723`
- `lag_07__CT3__flash_duration`: contribution `+0.010309`
- `lag_05__CT5__duck_amount`: contribution `+0.008000`
- `lag_12__CT_place_PALACEINTERIOR`: contribution `+0.007887`

Top utility-only movements:
- `lag_07__CT3__flash_duration`: contribution `+0.010309`
- `lag_10__T1__flash_duration`: contribution `+0.005152`

### tick `116278`, seconds `37.00`, LSTM delta `-0.2140`

Top all feature movements:
- `lag_02__T_place_TRUCK`: contribution `-0.041273`
- `lag_04__T_place_TRUCK`: contribution `-0.018283`
- `lag_11__CT_place_JUNGLE`: contribution `-0.011231`
- `lag_05__CT5__duck_amount`: contribution `-0.008000`
- `lag_12__CT_place_PALACEINTERIOR`: contribution `-0.007887`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.007120`
- `lag_11__T4__flash_duration`: contribution `-0.003833`

### tick `116182`, seconds `35.50`, LSTM delta `+0.2006`

Top all feature movements:
- `lag_01__T_place_TRUCK`: contribution `+0.033516`
- `lag_13__T_place_TRUCK`: contribution `+0.032464`
- `lag_08__T4__flash_duration`: contribution `+0.009735`
- `lag_00__T4__flash_duration`: contribution `+0.008682`
- `lag_08__CT_place_JUNGLE`: contribution `+0.008437`

Top utility-only movements:
- `lag_08__T4__flash_duration`: contribution `+0.009735`
- `lag_00__T4__flash_duration`: contribution `+0.008682`
- `lag_08__T_flash_duration_sum`: contribution `+0.002616`

### tick `115318`, seconds `22.00`, LSTM delta `-0.1464`

Top all feature movements:
- `lag_11__CT_place_STAIRS`: contribution `-0.009760`
- `lag_05__CT5__duck_amount`: contribution `-0.008000`
- `lag_11__CT3__flash_duration`: contribution `-0.006603`
- `lag_05__CT_place_SNIPERSNEST`: contribution `-0.006096`
- `lag_00__kill_diff_last_3s`: contribution `-0.006001`

Top utility-only movements:
- `lag_11__CT3__flash_duration`: contribution `-0.006603`
- `lag_00__CT3__flash_duration`: contribution `-0.003206`

### tick `115382`, seconds `23.00`, LSTM delta `-0.0879`

Top all feature movements:
- `lag_10__CT_place_UNDERPASS`: contribution `-0.007727`
- `lag_14__CT_place_STAIRS`: contribution `-0.006658`
- `lag_02__CT_place_SHOP`: contribution `-0.006402`
- `lag_00__CT_kills_last_3s`: contribution `-0.006113`
- `lag_00__kill_diff_last_3s`: contribution `-0.006001`

Top utility-only movements:
- `lag_13__CT3__flash_duration`: contribution `-0.005069`
- `lag_02__CT3__flash_duration`: contribution `-0.003484`
