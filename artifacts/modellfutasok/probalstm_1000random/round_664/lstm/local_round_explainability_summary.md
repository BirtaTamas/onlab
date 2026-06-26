# Local Round Explainability

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `7`

## Largest probability jumps

- tick `41947`, seconds `6.00`, LSTM `0.3318`, delta `-0.1225`
- tick `42651`, seconds `17.00`, LSTM `0.0487`, delta `-0.1149`
- tick `41979`, seconds `6.50`, LSTM `0.2535`, delta `-0.0783`
- tick `42043`, seconds `7.50`, LSTM `0.1768`, delta `-0.0486`
- tick `42075`, seconds `8.00`, LSTM `0.1311`, delta `-0.0457`
- tick `42267`, seconds `11.00`, LSTM `0.1664`, delta `+0.0429`
- tick `42107`, seconds `8.50`, LSTM `0.0994`, delta `-0.0317`
- tick `42011`, seconds `7.00`, LSTM `0.2254`, delta `-0.0281`
- tick `42299`, seconds `11.50`, LSTM `0.1939`, delta `+0.0275`
- tick `42555`, seconds `15.50`, LSTM `0.1671`, delta `-0.0254`

## Top 15 local ridge features

- `lag_14__CT_place_HOLE`: coefficient `0.001252`, |coef| `0.001252`
- `lag_05__T_place_OUTSIDETUNNEL`: coefficient `-0.001152`, |coef| `0.001152`
- `lag_03__T_place_OUTSIDETUNNEL`: coefficient `-0.001137`, |coef| `0.001137`
- `lag_04__T_place_OUTSIDETUNNEL`: coefficient `-0.001125`, |coef| `0.001125`
- `lag_06__CT_place_MIDDOORS`: coefficient `-0.001027`, |coef| `0.001027`
- `lag_00__T_kills_last_3s`: coefficient `-0.000994`, |coef| `0.000994`
- `lag_08__CT_place_LONGDOORS`: coefficient `-0.000940`, |coef| `0.000940`
- `lag_06__T_place_OUTSIDETUNNEL`: coefficient `-0.000926`, |coef| `0.000926`
- `lag_08__T_place_OUTSIDETUNNEL`: coefficient `-0.000861`, |coef| `0.000861`
- `lag_00__T_damage_last_5s`: coefficient `-0.000827`, |coef| `0.000827`
- `lag_00__kill_diff_last_3s`: coefficient `0.000794`, |coef| `0.000794`
- `lag_11__CT_place_SHORTSTAIRS`: coefficient `0.000791`, |coef| `0.000791`
- `lag_11__CT_place_CATWALK`: coefficient `-0.000767`, |coef| `0.000767`
- `lag_00__damage_diff_last_5s`: coefficient `0.000755`, |coef| `0.000755`
- `lag_07__T_place_OUTSIDETUNNEL`: coefficient `-0.000749`, |coef| `0.000749`

## Top 10 utility ridge features

- `lag_00__CT4__molly`: coefficient `0.000635` (raises CT win probability)
- `lag_04__CT_B_site_active_infernos`: coefficient `0.000612` (raises CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `-0.000602` (lowers CT win probability)
- `lag_00__CT4__smoke`: coefficient `0.000562` (raises CT win probability)
- `lag_03__CT_active_infernos`: coefficient `0.000539` (raises CT win probability)
- `lag_00__CT1__utility_total`: coefficient `0.000530` (raises CT win probability)
- `lag_12__CT3__molly`: coefficient `-0.000505` (lowers CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000497` (raises CT win probability)
- `lag_00__CT1__flash`: coefficient `0.000484` (raises CT win probability)
- `lag_15__CT_active_infernos`: coefficient `-0.000482` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_HOLE`: coefficient `0.001252` (raises CT win probability)
- `lag_05__T_place_OUTSIDETUNNEL`: coefficient `-0.001152` (lowers CT win probability)
- `lag_03__T_place_OUTSIDETUNNEL`: coefficient `-0.001137` (lowers CT win probability)
- `lag_04__T_place_OUTSIDETUNNEL`: coefficient `-0.001125` (lowers CT win probability)
- `lag_06__CT_place_MIDDOORS`: coefficient `-0.001027` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.000994` (lowers CT win probability)
- `lag_08__CT_place_LONGDOORS`: coefficient `-0.000940` (lowers CT win probability)
- `lag_06__T_place_OUTSIDETUNNEL`: coefficient `-0.000926` (lowers CT win probability)
- `lag_08__T_place_OUTSIDETUNNEL`: coefficient `-0.000861` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000827` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `41947`, seconds `6.00`, LSTM delta `-0.1225`

Top all feature movements:
- `lag_05__T_place_OUTSIDETUNNEL`: contribution `-0.005755`
- `lag_03__T_place_OUTSIDETUNNEL`: contribution `-0.005685`
- `lag_04__T_place_OUTSIDETUNNEL`: contribution `-0.005624`
- `lag_05__CT_place_MIDDOORS`: contribution `-0.003777`
- `lag_00__T_kills_last_3s`: contribution `-0.003148`

Top utility-only movements:
- `lag_00__CT4__molly`: contribution `-0.001563`

### tick `42651`, seconds `17.00`, LSTM delta `-0.1149`

Top all feature movements:
- `lag_14__CT_place_HOLE`: contribution `-0.013978`
- `lag_03__T_place_OUTSIDETUNNEL`: contribution `-0.005685`
- `lag_11__CT_place_SHORTSTAIRS`: contribution `-0.004409`
- `lag_08__CT_place_LONGDOORS`: contribution `-0.004118`
- `lag_00__T_kills_last_3s`: contribution `-0.003148`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `-0.002101`
- `lag_15__CT_B_site_active_infernos`: contribution `-0.002067`

### tick `41979`, seconds `6.50`, LSTM delta `-0.0783`

Top all feature movements:
- `lag_06__CT_place_MIDDOORS`: contribution `-0.005930`
- `lag_05__T_place_OUTSIDETUNNEL`: contribution `-0.005755`
- `lag_04__T_place_OUTSIDETUNNEL`: contribution `-0.005624`
- `lag_06__T_place_OUTSIDETUNNEL`: contribution `-0.004627`
- `lag_08__T5__is_scoped`: contribution `-0.002309`

Top utility-only movements:
- `lag_13__CT_molly_inv`: contribution `-0.000780`
- `lag_13__CT3__molly`: contribution `-0.000736`

### tick `42043`, seconds `7.50`, LSTM delta `-0.0486`

Top all feature movements:
- `lag_06__T_place_OUTSIDETUNNEL`: contribution `-0.004627`
- `lag_08__T_place_OUTSIDETUNNEL`: contribution `-0.004305`
- `lag_07__T_place_OUTSIDETUNNEL`: contribution `-0.003745`
- `lag_08__CT_place_MIDDOORS`: contribution `-0.002728`
- `lag_09__CT_place_MIDDOORS`: contribution `-0.001500`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `-0.000897`

### tick `42075`, seconds `8.00`, LSTM delta `-0.0457`

Top all feature movements:
- `lag_08__T_place_OUTSIDETUNNEL`: contribution `-0.004305`
- `lag_07__T_place_OUTSIDETUNNEL`: contribution `-0.003745`
- `lag_09__T_place_OUTSIDETUNNEL`: contribution `-0.003482`
- `lag_09__CT_place_MIDDOORS`: contribution `-0.002999`
- `lag_02__CT_place_BDOORS`: contribution `-0.002107`

Top utility-only movements:
- `lag_00__CT2__molly`: contribution `-0.001008`
