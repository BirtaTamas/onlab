# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-3dmax-bo3-Oe166BQltZjvHlE8qlepgF/furia-vs-3dmax-m1-nuke.csv`
- round_num: `9`

## Largest probability jumps

- tick `65229`, seconds `20.50`, LSTM `0.8795`, delta `+0.1266`
- tick `65197`, seconds `20.00`, LSTM `0.7529`, delta `+0.0894`
- tick `65293`, seconds `21.50`, LSTM `0.9484`, delta `+0.0518`
- tick `68493`, seconds `71.50`, LSTM `0.9333`, delta `-0.0416`
- tick `69485`, seconds `87.00`, LSTM `0.9049`, delta `+0.0388`
- tick `68557`, seconds `72.50`, LSTM `0.8859`, delta `-0.0345`
- tick `69197`, seconds `82.50`, LSTM `0.8749`, delta `-0.0343`
- tick `65069`, seconds `18.00`, LSTM `0.6348`, delta `+0.0262`
- tick `68653`, seconds `74.00`, LSTM `0.8818`, delta `+0.0251`
- tick `69517`, seconds `87.50`, LSTM `0.9294`, delta `+0.0245`

## Top 15 local ridge features

- `lag_05__T_place_CONTROL`: coefficient `0.000905`, |coef| `0.000905`
- `lag_00__T_place_HEAVEN`: coefficient `-0.000857`, |coef| `0.000857`
- `lag_07__T_place_CONTROL`: coefficient `0.000825`, |coef| `0.000825`
- `lag_06__T_place_CONTROL`: coefficient `0.000820`, |coef| `0.000820`
- `lag_03__T_place_CONTROL`: coefficient `0.000790`, |coef| `0.000790`
- `lag_06__T_place_VENDING`: coefficient `-0.000732`, |coef| `0.000732`
- `lag_07__CT_place_SQUEAKY`: coefficient `-0.000706`, |coef| `0.000706`
- `lag_04__T_place_CONTROL`: coefficient `0.000663`, |coef| `0.000663`
- `lag_00__kill_diff_last_3s`: coefficient `0.000621`, |coef| `0.000621`
- `lag_07__T_place_VENDING`: coefficient `-0.000620`, |coef| `0.000620`
- `lag_09__T_place_ADMIN`: coefficient `-0.000616`, |coef| `0.000616`
- `lag_00__damage_diff_last_5s`: coefficient `0.000612`, |coef| `0.000612`
- `lag_00__CT_shots_fired_sum`: coefficient `0.000590`, |coef| `0.000590`
- `lag_03__T_place_TROPHY`: coefficient `-0.000583`, |coef| `0.000583`
- `lag_00__T_place_CONTROL`: coefficient `-0.000566`, |coef| `0.000566`

## Top 10 utility ridge features

- `lag_00__T3__flash_duration`: coefficient `0.000497` (raises CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000397` (lowers CT win probability)
- `lag_00__T1__molly`: coefficient `-0.000340` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000331` (lowers CT win probability)
- `lag_00__T2__molly`: coefficient `-0.000330` (lowers CT win probability)
- `lag_02__T_flashes_last_5s`: coefficient `-0.000327` (lowers CT win probability)
- `lag_09__CT_B_site_active_infernos`: coefficient `-0.000322` (lowers CT win probability)
- `lag_01__T_flashes_last_5s`: coefficient `-0.000304` (lowers CT win probability)
- `lag_03__T_flashes_last_5s`: coefficient `-0.000294` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `0.000293` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_05__T_place_CONTROL`: coefficient `0.000905` (raises CT win probability)
- `lag_00__T_place_HEAVEN`: coefficient `-0.000857` (lowers CT win probability)
- `lag_07__T_place_CONTROL`: coefficient `0.000825` (raises CT win probability)
- `lag_06__T_place_CONTROL`: coefficient `0.000820` (raises CT win probability)
- `lag_03__T_place_CONTROL`: coefficient `0.000790` (raises CT win probability)
- `lag_06__T_place_VENDING`: coefficient `-0.000732` (lowers CT win probability)
- `lag_07__CT_place_SQUEAKY`: coefficient `-0.000706` (lowers CT win probability)
- `lag_04__T_place_CONTROL`: coefficient `0.000663` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.000621` (raises CT win probability)
- `lag_07__T_place_VENDING`: coefficient `-0.000620` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `65229`, seconds `20.50`, LSTM delta `+0.1266`

Top all feature movements:
- `lag_05__T_place_CONTROL`: contribution `+0.006434`
- `lag_07__T_place_CONTROL`: contribution `+0.005859`
- `lag_03__T_place_CONTROL`: contribution `+0.005613`
- `lag_06__CT_place_CONTROL`: contribution `+0.004594`
- `lag_00__T_place_CONTROL`: contribution `+0.004022`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `+0.003324`

### tick `65197`, seconds `20.00`, LSTM delta `+0.0894`

Top all feature movements:
- `lag_06__T_place_CONTROL`: contribution `+0.005827`
- `lag_05__CT_place_CONTROL`: contribution `+0.004859`
- `lag_04__T_place_CONTROL`: contribution `+0.004709`
- `lag_00__T_place_CONTROL`: contribution `+0.004022`
- `lag_02__T_place_CONTROL`: contribution `+0.003767`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `65293`, seconds `21.50`, LSTM delta `+0.0518`

Top all feature movements:
- `lag_00__T_place_CONTROL`: contribution `+0.008043`
- `lag_05__T_place_CONTROL`: contribution `+0.006434`
- `lag_07__T_place_CONTROL`: contribution `+0.005859`
- `lag_03__T_place_CONTROL`: contribution `-0.005613`
- `lag_02__T_place_CONTROL`: contribution `-0.003767`

Top utility-only movements:
- `lag_00__T3__flash_duration`: contribution `-0.003324`
- `lag_00__T_flash_duration_sum`: contribution `-0.001116`

### tick `68493`, seconds `71.50`, LSTM delta `-0.0416`

Top all feature movements:
- `lag_06__CT_place_VENDING`: contribution `-0.004202`
- `lag_11__CT_place_VENDING`: contribution `-0.004004`
- `lag_06__CT_place_TROPHY`: contribution `-0.003065`
- `lag_00__CT_place_RAFTERS`: contribution `-0.002391`
- `lag_11__CT_place_LOBBY`: contribution `-0.002305`

Top utility-only movements:
- `lag_02__CT3__smoke`: contribution `-0.000461`
- `lag_07__CT5__smoke`: contribution `-0.000441`

### tick `69485`, seconds `87.00`, LSTM delta `+0.0388`

Top all feature movements:
- `lag_09__T_place_ADMIN`: contribution `+0.011979`
- `lag_07__CT_place_SQUEAKY`: contribution `+0.009388`
- `lag_10__CT_place_SQUEAKY`: contribution `+0.004859`
- `lag_05__CT_place_VENTS`: contribution `+0.004179`
- `lag_00__T_place_CONTROL`: contribution `-0.004022`

Top utility-only movements:
- `lag_08__CT5__molly`: contribution `+0.000229`
