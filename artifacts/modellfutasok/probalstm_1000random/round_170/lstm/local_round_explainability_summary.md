# Local Round Explainability

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `16`

## Largest probability jumps

- tick `139059`, seconds `20.00`, LSTM `0.1295`, delta `-0.0520`
- tick `137811`, seconds `0.50`, LSTM `0.0879`, delta `-0.0510`
- tick `138707`, seconds `14.50`, LSTM `0.2563`, delta `+0.0443`
- tick `138963`, seconds `18.50`, LSTM `0.1765`, delta `-0.0387`
- tick `138067`, seconds `4.50`, LSTM `0.2022`, delta `+0.0311`
- tick `140243`, seconds `38.50`, LSTM `0.0833`, delta `-0.0265`
- tick `138323`, seconds `8.50`, LSTM `0.1607`, delta `+0.0252`
- tick `137875`, seconds `1.50`, LSTM `0.1072`, delta `+0.0249`
- tick `140211`, seconds `38.00`, LSTM `0.1099`, delta `-0.0241`
- tick `138131`, seconds `5.50`, LSTM `0.1865`, delta `-0.0224`

## Top 15 local ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001037`, |coef| `0.001037`
- `lag_00__CT_place_TUNNEL`: coefficient `-0.000664`, |coef| `0.000664`
- `lag_08__T_place_STREET`: coefficient `0.000561`, |coef| `0.000561`
- `lag_02__CT_flashes_last_5s`: coefficient `0.000522`, |coef| `0.000522`
- `lag_06__T_place_STREET`: coefficient `0.000494`, |coef| `0.000494`
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.000476`, |coef| `0.000476`
- `lag_10__CT_place_TUNNELSTAIRS`: coefficient `-0.000459`, |coef| `0.000459`
- `lag_02__CT_place_TUNNEL`: coefficient `-0.000444`, |coef| `0.000444`
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `0.000441`, |coef| `0.000441`
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.000434`, |coef| `0.000434`
- `lag_00__T_place_STREET`: coefficient `-0.000420`, |coef| `0.000420`
- `lag_03__T_place_WALKWAY`: coefficient `-0.000408`, |coef| `0.000408`
- `lag_03__T_place_TSTAIRS`: coefficient `0.000393`, |coef| `0.000393`
- `lag_08__T_place_TSTAIRS`: coefficient `-0.000393`, |coef| `0.000393`
- `lag_07__T_place_STREET`: coefficient `0.000381`, |coef| `0.000381`

## Top 10 utility ridge features

- `lag_02__CT_flashes_last_5s`: coefficient `0.000522` (raises CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `0.000329` (raises CT win probability)
- `lag_02__T5__flash_duration`: coefficient `-0.000326` (lowers CT win probability)
- `lag_11__CT3__flash_duration`: coefficient `-0.000313` (lowers CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.000292` (lowers CT win probability)
- `lag_03__CT_A_site_active_infernos`: coefficient `0.000290` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000278` (raises CT win probability)
- `lag_14__CT3__flash_duration`: coefficient `-0.000267` (lowers CT win probability)
- `lag_15__T5__flash_duration`: coefficient `-0.000263` (lowers CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000255` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_place_CTSIDEUPPER`: coefficient `-0.001037` (lowers CT win probability)
- `lag_00__CT_place_TUNNEL`: coefficient `-0.000664` (lowers CT win probability)
- `lag_08__T_place_STREET`: coefficient `0.000561` (raises CT win probability)
- `lag_06__T_place_STREET`: coefficient `0.000494` (raises CT win probability)
- `lag_00__CT_place_LOWERTUNNEL`: coefficient `-0.000476` (lowers CT win probability)
- `lag_10__CT_place_TUNNELSTAIRS`: coefficient `-0.000459` (lowers CT win probability)
- `lag_02__CT_place_TUNNEL`: coefficient `-0.000444` (lowers CT win probability)
- `lag_03__CT_place_LOWERTUNNEL`: coefficient `0.000441` (raises CT win probability)
- `lag_00__CT_place_TUNNELSTAIRS`: coefficient `0.000434` (raises CT win probability)
- `lag_00__T_place_STREET`: coefficient `-0.000420` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `139059`, seconds `20.00`, LSTM delta `-0.0520`

Top all feature movements:
- `lag_00__CT_place_TUNNEL`: contribution `-0.010663`
- `lag_00__CT_place_TUNNELSTAIRS`: contribution `-0.006106`
- `lag_02__CT_place_TUNNELSTAIRS`: contribution `-0.003120`
- `lag_05__T5__flash_duration`: contribution `-0.002201`
- `lag_14__CT3__flash_duration`: contribution `-0.001757`

Top utility-only movements:
- `lag_05__T5__flash_duration`: contribution `-0.002201`
- `lag_14__CT3__flash_duration`: contribution `-0.001757`
- `lag_02__CT3__flash_duration`: contribution `-0.001595`
- `lag_03__CT_A_site_active_infernos`: contribution `-0.001022`
- `lag_05__CT1__flash_duration`: contribution `-0.000828`

### tick `137811`, seconds `0.50`, LSTM delta `-0.0510`

Top all feature movements:
- `lag_01__CT_place_CTSIDEUPPER`: contribution `-0.026717`
- `lag_01__T_place_TSPAWN`: contribution `-0.000910`
- `lag_00__CT_flashes_last_5s`: contribution `-0.000652`
- `lag_01__T_closest_enemy_dist`: contribution `-0.000618`
- `lag_00__T_velocity_mean`: contribution `-0.000527`

Top utility-only movements:
- `lag_00__CT_flashes_last_5s`: contribution `-0.000652`
- `lag_01__flash_inv_diff`: contribution `-0.000283`
- `lag_01__T4__flash`: contribution `-0.000166`
- `lag_01__CT1__molly`: contribution `-0.000165`
- `lag_01__CT3__smoke`: contribution `-0.000164`

### tick `138707`, seconds `14.50`, LSTM delta `+0.0443`

Top all feature movements:
- `lag_10__CT_place_TUNNELSTAIRS`: contribution `+0.006464`
- `lag_12__CT_place_TUNNELSTAIRS`: contribution `+0.004095`
- `lag_08__T_place_STREET`: contribution `-0.003082`
- `lag_08__CT_place_MAIN`: contribution `+0.002533`
- `lag_00__T_place_STREET`: contribution `+0.002310`

Top utility-only movements:
- `lag_03__CT3__flash_duration`: contribution `+0.002169`
- `lag_03__CT_A_site_active_infernos`: contribution `+0.001022`

### tick `138963`, seconds `18.50`, LSTM delta `-0.0387`

Top all feature movements:
- `lag_08__T_place_STREET`: contribution `-0.003082`
- `lag_02__T5__flash_duration`: contribution `-0.002458`
- `lag_11__CT3__flash_duration`: contribution `-0.002061`
- `lag_02__CT1__flash_duration`: contribution `-0.001387`
- `lag_09__CT_place_CONNECTOR`: contribution `-0.001199`

Top utility-only movements:
- `lag_02__T5__flash_duration`: contribution `-0.002458`
- `lag_11__CT3__flash_duration`: contribution `-0.002061`
- `lag_02__CT1__flash_duration`: contribution `-0.001387`
- `lag_00__CT_A_site_active_infernos`: contribution `-0.000725`
- `lag_11__CT_A_site_active_infernos`: contribution `-0.000568`

### tick `138067`, seconds `4.50`, LSTM delta `+0.0311`

Top all feature movements:
- `lag_09__CT_place_CTSIDEUPPER`: contribution `+0.005989`
- `lag_04__CT_place_LOWERTUNNEL`: contribution `+0.003509`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.003500`
- `lag_06__CT_flashes_last_5s`: contribution `+0.001553`
- `lag_04__CT_place_PALACEINTERIOR`: contribution `+0.000883`

Top utility-only movements:
- `lag_06__CT_flashes_last_5s`: contribution `+0.001553`
- `lag_09__T3__utility_total`: contribution `+0.000265`
