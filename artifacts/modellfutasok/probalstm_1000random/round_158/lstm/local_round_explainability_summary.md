# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `3`

## Largest probability jumps

- tick `20676`, seconds `34.50`, LSTM `0.0988`, delta `-0.1641`
- tick `19140`, seconds `10.50`, LSTM `0.3500`, delta `+0.0994`
- tick `22916`, seconds `69.50`, LSTM `0.0255`, delta `-0.0953`
- tick `20708`, seconds `35.00`, LSTM `0.1940`, delta `+0.0953`
- tick `23140`, seconds `73.00`, LSTM `0.0819`, delta `-0.0801`
- tick `24388`, seconds `92.50`, LSTM `0.0580`, delta `-0.0776`
- tick `20772`, seconds `36.00`, LSTM `0.1029`, delta `-0.0757`
- tick `19428`, seconds `15.00`, LSTM `0.2762`, delta `-0.0740`
- tick `18948`, seconds `7.50`, LSTM `0.2736`, delta `-0.0734`
- tick `23108`, seconds `72.50`, LSTM `0.1620`, delta `+0.0621`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001960`, |coef| `0.001960`
- `lag_00__kill_diff_last_3s`: coefficient `0.001945`, |coef| `0.001945`
- `lag_00__T_kills_last_3s`: coefficient `-0.001824`, |coef| `0.001824`
- `lag_00__damage_diff_last_5s`: coefficient `0.001792`, |coef| `0.001792`
- `lag_00__T_damage_last_5s`: coefficient `-0.001540`, |coef| `0.001540`
- `lag_02__T_place_MIDDOORS`: coefficient `-0.001328`, |coef| `0.001328`
- `lag_00__T_place_SIDE`: coefficient `-0.001306`, |coef| `0.001306`
- `lag_05__CT_place_BDOORS`: coefficient `-0.001285`, |coef| `0.001285`
- `lag_06__CT1__duck_amount`: coefficient `-0.001272`, |coef| `0.001272`
- `lag_14__CT_flashed_players`: coefficient `-0.001235`, |coef| `0.001235`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001210`, |coef| `0.001210`
- `lag_00__T_place_PIT`: coefficient `-0.001188`, |coef| `0.001188`
- `lag_08__CT_place_EXTENDEDA`: coefficient `-0.001160`, |coef| `0.001160`
- `lag_00__T3__is_walking`: coefficient `-0.001154`, |coef| `0.001154`
- `lag_01__T3__is_walking`: coefficient `0.001153`, |coef| `0.001153`

## Top 10 utility ridge features

- `lag_04__T_A_site_active_infernos`: coefficient `-0.000937` (lowers CT win probability)
- `lag_00__CT5__smoke`: coefficient `0.000706` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000697` (raises CT win probability)
- `lag_00__CT_flashes_last_5s`: coefficient `0.000647` (raises CT win probability)
- `lag_03__CT_flashes_last_5s`: coefficient `0.000637` (raises CT win probability)
- `lag_00__CT_smoke_inv`: coefficient `0.000636` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000631` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000606` (raises CT win probability)
- `lag_08__CT_flashes_last_5s`: coefficient `-0.000550` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `0.000545` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001960` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001945` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001824` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001792` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001540` (lowers CT win probability)
- `lag_02__T_place_MIDDOORS`: coefficient `-0.001328` (lowers CT win probability)
- `lag_00__T_place_SIDE`: coefficient `-0.001306` (lowers CT win probability)
- `lag_05__CT_place_BDOORS`: coefficient `-0.001285` (lowers CT win probability)
- `lag_06__CT1__duck_amount`: coefficient `-0.001272` (lowers CT win probability)
- `lag_14__CT_flashed_players`: coefficient `-0.001235` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `20676`, seconds `34.50`, LSTM delta `-0.1641`

Top all feature movements:
- `lag_14__CT_flashed_players`: contribution `-0.008111`
- `lag_08__CT_place_EXTENDEDA`: contribution `-0.006512`
- `lag_15__CT_place_EXTENDEDA`: contribution `-0.006009`
- `lag_00__T_kills_last_3s`: contribution `-0.005780`
- `lag_06__CT1__duck_amount`: contribution `-0.004854`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.002788`

### tick `19140`, seconds `10.50`, LSTM delta `+0.0994`

Top all feature movements:
- `lag_05__CT_place_HOLE`: contribution `+0.012670`
- `lag_05__CT_place_BDOORS`: contribution `+0.012359`
- `lag_04__CT_place_HOLE`: contribution `+0.010556`
- `lag_00__CT_shots_fired_sum`: contribution `+0.008169`
- `lag_00__damage_diff_last_5s`: contribution `+0.003518`

Top utility-only movements:
- `lag_02__CT_B_site_active_infernos`: contribution `+0.001873`
- `lag_06__CT1__flash_duration`: contribution `+0.001463`

### tick `22916`, seconds `69.50`, LSTM delta `-0.0953`

Top all feature movements:
- `lag_00__T_kills_last_3s`: contribution `-0.005780`
- `lag_10__CT_place_EXTENDEDA`: contribution `-0.005670`
- `lag_02__T_place_MIDDOORS`: contribution `-0.005644`
- `lag_00__kill_diff_last_3s`: contribution `-0.004681`
- `lag_00__damage_diff_last_5s`: contribution `-0.004044`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `20708`, seconds `35.00`, LSTM delta `+0.0953`

Top all feature movements:
- `lag_01__CT_place_UPPERTUNNEL`: contribution `+0.007405`
- `lag_00__kill_diff_last_3s`: contribution `+0.004681`
- `lag_00__T_shots_fired_sum`: contribution `+0.004537`
- `lag_12__CT1__duck_amount`: contribution `+0.004167`
- `lag_00__CT1__duck_amount`: contribution `+0.003465`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `23140`, seconds `73.00`, LSTM delta `-0.0801`

Top all feature movements:
- `lag_00__T_place_SIDE`: contribution `-0.025269`
- `lag_00__CT_place_EXTENDEDA`: contribution `-0.006445`
- `lag_04__CT_flashes_last_5s`: contribution `-0.004938`
- `lag_00__kill_diff_last_3s`: contribution `-0.004681`
- `lag_01__T3__is_walking`: contribution `-0.002677`

Top utility-only movements:
- `lag_04__CT_flashes_last_5s`: contribution `-0.004938`
