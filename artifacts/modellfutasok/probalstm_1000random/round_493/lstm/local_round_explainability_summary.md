# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `5`

## Largest probability jumps

- tick `38419`, seconds `79.50`, LSTM `0.3862`, delta `+0.2609`
- tick `34771`, seconds `22.50`, LSTM `0.3156`, delta `-0.1828`
- tick `38899`, seconds `87.00`, LSTM `0.6966`, delta `+0.1469`
- tick `38451`, seconds `80.00`, LSTM `0.5000`, delta `+0.1138`
- tick `39027`, seconds `89.00`, LSTM `0.8942`, delta `+0.0987`
- tick `38995`, seconds `88.50`, LSTM `0.7955`, delta `+0.0654`
- tick `38291`, seconds `77.50`, LSTM `0.1839`, delta `-0.0653`
- tick `38483`, seconds `80.50`, LSTM `0.5615`, delta `+0.0615`
- tick `34963`, seconds `25.50`, LSTM `0.2400`, delta `+0.0581`
- tick `34835`, seconds `23.50`, LSTM `0.2304`, delta `-0.0573`

## Top 15 local ridge features

- `lag_00__CT_place_HOLE`: coefficient `-0.002889`, |coef| `0.002889`
- `lag_10__T_place_EXTENDEDA`: coefficient `0.002687`, |coef| `0.002687`
- `lag_00__kill_diff_last_3s`: coefficient `0.002319`, |coef| `0.002319`
- `lag_00__T_place_BDOORS`: coefficient `-0.002249`, |coef| `0.002249`
- `lag_09__T_place_EXTENDEDA`: coefficient `0.002017`, |coef| `0.002017`
- `lag_11__T4__flash_duration`: coefficient `0.001858`, |coef| `0.001858`
- `lag_01__damage_diff_last_5s`: coefficient `0.001813`, |coef| `0.001813`
- `lag_00__CT_kills_last_3s`: coefficient `0.001796`, |coef| `0.001796`
- `lag_00__damage_diff_last_5s`: coefficient `0.001721`, |coef| `0.001721`
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `-0.001618`, |coef| `0.001618`
- `lag_12__T_place_EXTENDEDA`: coefficient `0.001561`, |coef| `0.001561`
- `lag_06__T_place_EXTENDEDA`: coefficient `0.001497`, |coef| `0.001497`
- `lag_00__CT_damage_last_5s`: coefficient `0.001475`, |coef| `0.001475`
- `lag_13__T_place_EXTENDEDA`: coefficient `0.001455`, |coef| `0.001455`
- `lag_15__T1__flash_duration`: coefficient `0.001440`, |coef| `0.001440`

## Top 10 utility ridge features

- `lag_11__T4__flash_duration`: coefficient `0.001858` (raises CT win probability)
- `lag_15__T1__flash_duration`: coefficient `0.001440` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `-0.001381` (lowers CT win probability)
- `lag_15__T_flash_duration_sum`: coefficient `0.001358` (raises CT win probability)
- `lag_09__T4__flash_duration`: coefficient `0.001190` (raises CT win probability)
- `lag_02__T1__flash_duration`: coefficient `-0.001184` (lowers CT win probability)
- `lag_15__T3__flash_duration`: coefficient `0.001101` (raises CT win probability)
- `lag_13__T4__flash_duration`: coefficient `0.001035` (raises CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.001020` (raises CT win probability)
- `lag_10__T4__flash_duration`: coefficient `0.001007` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_place_HOLE`: coefficient `-0.002889` (lowers CT win probability)
- `lag_10__T_place_EXTENDEDA`: coefficient `0.002687` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002319` (raises CT win probability)
- `lag_00__T_place_BDOORS`: coefficient `-0.002249` (lowers CT win probability)
- `lag_09__T_place_EXTENDEDA`: coefficient `0.002017` (raises CT win probability)
- `lag_01__damage_diff_last_5s`: coefficient `0.001813` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001796` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001721` (raises CT win probability)
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `-0.001618` (lowers CT win probability)
- `lag_12__T_place_EXTENDEDA`: coefficient `0.001561` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `38419`, seconds `79.50`, LSTM delta `+0.2609`

Top all feature movements:
- `lag_00__T_place_BDOORS`: contribution `+0.028125`
- `lag_01__T_place_BDOORS`: contribution `+0.014936`
- `lag_15__T1__flash_duration`: contribution `+0.010979`
- `lag_01__T3__flash_duration`: contribution `+0.010867`
- `lag_15__T_flash_duration_sum`: contribution `+0.009053`

Top utility-only movements:
- `lag_15__T1__flash_duration`: contribution `+0.010979`
- `lag_01__T3__flash_duration`: contribution `+0.010867`
- `lag_15__T_flash_duration_sum`: contribution `+0.009053`
- `lag_02__T1__flash_duration`: contribution `+0.009025`
- `lag_15__T3__flash_duration`: contribution `+0.008666`

### tick `34771`, seconds `22.50`, LSTM delta `-0.1828`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `-0.032258`
- `lag_11__T4__flash_duration`: contribution `-0.013334`
- `lag_05__CT_place_SHORTSTAIRS`: contribution `-0.009021`
- `lag_09__T1__is_scoped`: contribution `-0.007673`
- `lag_00__kill_diff_last_3s`: contribution `-0.005581`

Top utility-only movements:
- `lag_11__T4__flash_duration`: contribution `-0.013334`
- `lag_00__CT_B_site_active_infernos`: contribution `-0.002975`

### tick `38899`, seconds `87.00`, LSTM delta `+0.1469`

Top all feature movements:
- `lag_15__T_place_BDOORS`: contribution `+0.015705`
- `lag_10__T_place_EXTENDEDA`: contribution `+0.013321`
- `lag_14__CT_place_HOLE`: contribution `+0.011445`
- `lag_12__T_place_EXTENDEDA`: contribution `+0.007738`
- `lag_06__T_place_EXTENDEDA`: contribution `+0.007423`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `38451`, seconds `80.00`, LSTM delta `+0.1138`

Top all feature movements:
- `lag_00__CT_place_HOLE`: contribution `+0.032258`
- `lag_01__T_place_BDOORS`: contribution `-0.014936`
- `lag_02__T_place_BDOORS`: contribution `+0.006942`
- `lag_03__CT_place_HOLE`: contribution `+0.004848`
- `lag_01__damage_diff_last_5s`: contribution `+0.004090`

Top utility-only movements:
- `lag_02__T3__flash_duration`: contribution `+0.003773`
- `lag_03__T1__flash_duration`: contribution `+0.003259`
- `lag_10__T4__flash_duration`: contribution `+0.003132`

### tick `39027`, seconds `89.00`, LSTM delta `+0.0987`

Top all feature movements:
- `lag_10__T_place_EXTENDEDA`: contribution `+0.013321`
- `lag_14__T_place_EXTENDEDA`: contribution `+0.006223`
- `lag_00__kill_diff_last_3s`: contribution `+0.005581`
- `lag_00__T_place_EXTENDEDA`: contribution `+0.005545`
- `lag_00__CT_kills_last_3s`: contribution `+0.005186`

Top utility-only movements:
- No utility movement among the top local contributors.
