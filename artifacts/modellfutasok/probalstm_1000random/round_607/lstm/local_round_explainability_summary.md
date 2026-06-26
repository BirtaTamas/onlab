# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-falcons-vs-mouz-bo3-plkh_Ps38mI3o_rFlgAljz/falcons-vs-mouz-m3-nuke-p3.csv`
- round_num: `6`

## Largest probability jumps

- tick `57836`, seconds `31.00`, LSTM `0.5317`, delta `+0.1367`
- tick `58124`, seconds `35.50`, LSTM `0.7615`, delta `+0.1309`
- tick `61580`, seconds `89.50`, LSTM `0.9101`, delta `+0.0971`
- tick `55884`, seconds `0.50`, LSTM `0.1056`, delta `-0.0772`
- tick `58092`, seconds `35.00`, LSTM `0.6306`, delta `+0.0492`
- tick `56524`, seconds `10.50`, LSTM `0.2405`, delta `+0.0446`
- tick `61260`, seconds `84.50`, LSTM `0.8530`, delta `-0.0399`
- tick `58636`, seconds `43.50`, LSTM `0.7380`, delta `-0.0378`
- tick `61708`, seconds `91.50`, LSTM `0.9661`, delta `+0.0376`
- tick `56876`, seconds `16.00`, LSTM `0.4306`, delta `+0.0343`

## Top 15 local ridge features

- `lag_15__CT_place_SECRET`: coefficient `0.001863`, |coef| `0.001863`
- `lag_01__T_place_SECRET`: coefficient `0.001680`, |coef| `0.001680`
- `lag_04__CT_place_LOCKERROOM`: coefficient `0.001618`, |coef| `0.001618`
- `lag_00__CT_kills_last_3s`: coefficient `0.001425`, |coef| `0.001425`
- `lag_00__CT_place_TROPHY`: coefficient `0.001273`, |coef| `0.001273`
- `lag_01__CT_place_TUNNELS`: coefficient `0.001223`, |coef| `0.001223`
- `lag_13__CT_place_OBSERVATION`: coefficient `0.001191`, |coef| `0.001191`
- `lag_00__kill_diff_last_3s`: coefficient `0.001188`, |coef| `0.001188`
- `lag_12__CT_place_GARAGE`: coefficient `0.001134`, |coef| `0.001134`
- `lag_02__T_place_SECRET`: coefficient `0.001123`, |coef| `0.001123`
- `lag_01__CT_place_SECRET`: coefficient `-0.001114`, |coef| `0.001114`
- `lag_00__CT_damage_last_5s`: coefficient `0.001081`, |coef| `0.001081`
- `lag_00__T4__flash`: coefficient `-0.001066`, |coef| `0.001066`
- `lag_00__T4__utility_total`: coefficient `-0.001059`, |coef| `0.001059`
- `lag_00__CT5__is_walking`: coefficient `-0.001049`, |coef| `0.001049`

## Top 10 utility ridge features

- `lag_00__T4__flash`: coefficient `-0.001066` (lowers CT win probability)
- `lag_00__T4__utility_total`: coefficient `-0.001059` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000855` (lowers CT win probability)
- `lag_00__T_molly_inv`: coefficient `-0.000767` (lowers CT win probability)
- `lag_01__T4__utility_total`: coefficient `-0.000761` (lowers CT win probability)
- `lag_00__utility_inv_diff`: coefficient `0.000745` (raises CT win probability)
- `lag_00__T_utility_inv`: coefficient `-0.000735` (lowers CT win probability)
- `lag_00__molly_inv_diff`: coefficient `0.000732` (raises CT win probability)
- `lag_01__T_smoke_inv`: coefficient `-0.000707` (lowers CT win probability)
- `lag_01__T4__flash`: coefficient `-0.000702` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_15__CT_place_SECRET`: coefficient `0.001863` (raises CT win probability)
- `lag_01__T_place_SECRET`: coefficient `0.001680` (raises CT win probability)
- `lag_04__CT_place_LOCKERROOM`: coefficient `0.001618` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001425` (raises CT win probability)
- `lag_00__CT_place_TROPHY`: coefficient `0.001273` (raises CT win probability)
- `lag_01__CT_place_TUNNELS`: coefficient `0.001223` (raises CT win probability)
- `lag_13__CT_place_OBSERVATION`: coefficient `0.001191` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001188` (raises CT win probability)
- `lag_12__CT_place_GARAGE`: coefficient `0.001134` (raises CT win probability)
- `lag_02__T_place_SECRET`: coefficient `0.001123` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `57836`, seconds `31.00`, LSTM delta `+0.1367`

Top all feature movements:
- `lag_15__CT_place_SECRET`: contribution `+0.019176`
- `lag_12__CT_place_GARAGE`: contribution `+0.008152`
- `lag_13__CT_place_GARAGE`: contribution `+0.004597`
- `lag_00__CT_kills_last_3s`: contribution `+0.004113`
- `lag_08__CT_place_GARAGE`: contribution `+0.003837`

Top utility-only movements:
- `lag_00__T4__flash`: contribution `+0.002896`
- `lag_00__T4__utility_total`: contribution `+0.002470`

### tick `58124`, seconds `35.50`, LSTM delta `+0.1309`

Top all feature movements:
- `lag_04__CT_place_LOCKERROOM`: contribution `+0.020140`
- `lag_01__CT_place_SECRET`: contribution `+0.011468`
- `lag_02__T_place_SECRET`: contribution `+0.005910`
- `lag_00__CT_kills_last_3s`: contribution `+0.004113`
- `lag_01__CT_place_TUNNELS`: contribution `+0.003742`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `61580`, seconds `89.50`, LSTM delta `+0.0971`

Top all feature movements:
- `lag_10__CT_place_VENDING`: contribution `+0.012581`
- `lag_04__CT_place_VENDING`: contribution `+0.008867`
- `lag_15__CT_place_TROPHY`: contribution `+0.008452`
- `lag_10__CT_place_TROPHY`: contribution `+0.007653`
- `lag_04__CT_place_LOBBY`: contribution `+0.006540`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `55884`, seconds `0.50`, LSTM delta `-0.0772`

Top all feature movements:
- `lag_01__CT_place_CTSPAWN`: contribution `-0.004841`
- `lag_01__T_place_TSPAWN`: contribution `-0.004375`
- `lag_01__CT_closest_enemy_dist`: contribution `-0.004222`
- `lag_01__T_closest_enemy_dist`: contribution `-0.004169`
- `lag_01__centroid_distance_xy`: contribution `-0.003643`

Top utility-only movements:
- `lag_01__utility_inv_diff`: contribution `-0.002113`
- `lag_01__molly_inv_diff`: contribution `-0.001700`
- `lag_01__T4__utility_total`: contribution `-0.001684`
- `lag_01__T_smoke_inv`: contribution `-0.001613`
- `lag_01__T_utility_inv`: contribution `-0.001454`

### tick `58092`, seconds `35.00`, LSTM delta `+0.0492`

Top all feature movements:
- `lag_03__CT_place_LOCKERROOM`: contribution `+0.012699`
- `lag_01__T_place_SECRET`: contribution `+0.008841`
- `lag_00__CT_place_SECRET`: contribution `+0.004966`
- `lag_04__CT_place_HELL`: contribution `-0.003088`
- `lag_01__CT4__is_scoped`: contribution `+0.002966`

Top utility-only movements:
- No utility movement among the top local contributors.
