# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `1`

## Largest probability jumps

- tick `2586`, seconds `20.50`, LSTM `0.1980`, delta `-0.2296`
- tick `2874`, seconds `25.00`, LSTM `0.0455`, delta `-0.1558`
- tick `2714`, seconds `22.50`, LSTM `0.2113`, delta `+0.1523`
- tick `2618`, seconds `21.00`, LSTM `0.1141`, delta `-0.0839`
- tick `2810`, seconds `24.00`, LSTM `0.2008`, delta `-0.0428`
- tick `2554`, seconds `20.00`, LSTM `0.4276`, delta `-0.0365`
- tick `2650`, seconds `21.50`, LSTM `0.0854`, delta `-0.0287`
- tick `2682`, seconds `22.00`, LSTM `0.0590`, delta `-0.0264`
- tick `2778`, seconds `23.50`, LSTM `0.2436`, delta `+0.0257`
- tick `2522`, seconds `19.50`, LSTM `0.4642`, delta `-0.0179`

## Top 15 local ridge features

- `lag_04__CT_place_TRUCK`: coefficient `0.002642`, |coef| `0.002642`
- `lag_03__CT_place_TRUCK`: coefficient `0.002315`, |coef| `0.002315`
- `lag_15__CT_place_TRUCK`: coefficient `-0.001554`, |coef| `0.001554`
- `lag_00__T_kills_last_3s`: coefficient `-0.001453`, |coef| `0.001453`
- `lag_02__T_place_PALACEINTERIOR`: coefficient `-0.001436`, |coef| `0.001436`
- `lag_13__CT_place_SNIPERSNEST`: coefficient `0.001432`, |coef| `0.001432`
- `lag_01__T_place_SCAFFOLDING`: coefficient `0.001429`, |coef| `0.001429`
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `0.001375`, |coef| `0.001375`
- `lag_00__CT4__alive`: coefficient `0.001355`, |coef| `0.001355`
- `lag_02__T5__duck_amount`: coefficient `0.001350`, |coef| `0.001350`
- `lag_14__T5__duck_amount`: coefficient `-0.001307`, |coef| `0.001307`
- `lag_00__CT3__is_walking`: coefficient `-0.001238`, |coef| `0.001238`
- `lag_00__kill_diff_last_3s`: coefficient `0.001220`, |coef| `0.001220`
- `lag_00__CT_place_CONNECTOR`: coefficient `0.001193`, |coef| `0.001193`
- `lag_04__T4__is_walking`: coefficient `0.001169`, |coef| `0.001169`

## Top 10 utility ridge features

- `lag_04__T_A_site_active_infernos`: coefficient `-0.001148` (lowers CT win probability)
- `lag_07__T5__molly`: coefficient `0.001055` (raises CT win probability)
- `lag_09__T5__smoke`: coefficient `0.001038` (raises CT win probability)
- `lag_04__T_active_infernos`: coefficient `-0.000784` (lowers CT win probability)
- `lag_08__T5__molly`: coefficient `0.000618` (raises CT win probability)
- `lag_10__T5__smoke`: coefficient `0.000590` (raises CT win probability)
- `lag_03__T_A_site_active_smokes`: coefficient `-0.000575` (lowers CT win probability)
- `lag_05__T_A_site_active_infernos`: coefficient `-0.000570` (lowers CT win probability)
- `lag_09__T5__utility_total`: coefficient `0.000564` (raises CT win probability)
- `lag_07__T5__utility_total`: coefficient `0.000549` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_04__CT_place_TRUCK`: coefficient `0.002642` (raises CT win probability)
- `lag_03__CT_place_TRUCK`: coefficient `0.002315` (raises CT win probability)
- `lag_15__CT_place_TRUCK`: coefficient `-0.001554` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001453` (lowers CT win probability)
- `lag_02__T_place_PALACEINTERIOR`: coefficient `-0.001436` (lowers CT win probability)
- `lag_13__CT_place_SNIPERSNEST`: coefficient `0.001432` (raises CT win probability)
- `lag_01__T_place_SCAFFOLDING`: coefficient `0.001429` (raises CT win probability)
- `lag_00__CT_place_PALACEINTERIOR`: coefficient `0.001375` (raises CT win probability)
- `lag_00__CT4__alive`: coefficient `0.001355` (raises CT win probability)
- `lag_02__T5__duck_amount`: coefficient `0.001350` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `2586`, seconds `20.50`, LSTM delta `-0.2296`

Top all feature movements:
- `lag_04__CT_place_TRUCK`: contribution `-0.017043`
- `lag_03__CT_place_TRUCK`: contribution `-0.014929`
- `lag_13__CT_place_SNIPERSNEST`: contribution `-0.007672`
- `lag_00__CT_place_PALACEINTERIOR`: contribution `-0.005604`
- `lag_02__T5__duck_amount`: contribution `-0.005127`

Top utility-only movements:
- `lag_04__T_A_site_active_infernos`: contribution `-0.003416`
- `lag_07__T5__molly`: contribution `-0.002333`

### tick `2874`, seconds `25.00`, LSTM delta `-0.1558`

Top all feature movements:
- `lag_01__T_place_SCAFFOLDING`: contribution `-0.048663`
- `lag_06__T_place_SCAFFOLDING`: contribution `-0.037454`
- `lag_04__T_place_SCAFFOLDING`: contribution `+0.005535`
- `lag_00__CT_place_SHOP`: contribution `-0.004684`
- `lag_00__T_kills_last_3s`: contribution `-0.004604`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `2714`, seconds `22.50`, LSTM delta `+0.1523`

Top all feature movements:
- `lag_01__T_place_SCAFFOLDING`: contribution `+0.048663`
- `lag_00__T_place_PALACEINTERIOR`: contribution `+0.004929`
- `lag_00__CT_place_SHOP`: contribution `+0.004684`
- `lag_08__CT_place_TRUCK`: contribution `+0.004466`
- `lag_00__CT_place_JUNGLE`: contribution `+0.003860`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `2618`, seconds `21.00`, LSTM delta `-0.0839`

Top all feature movements:
- `lag_04__CT_place_TRUCK`: contribution `-0.017043`
- `lag_05__CT_place_TRUCK`: contribution `-0.006291`
- `lag_04__CT_place_CONNECTOR`: contribution `-0.003818`
- `lag_14__CT_place_SNIPERSNEST`: contribution `-0.003048`
- `lag_03__T5__duck_amount`: contribution `-0.002851`

Top utility-only movements:
- `lag_05__T_A_site_active_infernos`: contribution `-0.001697`
- `lag_08__T5__molly`: contribution `-0.001368`

### tick `2810`, seconds `24.00`, LSTM delta `-0.0428`

Top all feature movements:
- `lag_01__T_place_SCAFFOLDING`: contribution `+0.048663`
- `lag_02__T_place_SCAFFOLDING`: contribution `-0.024553`
- `lag_04__T_place_SCAFFOLDING`: contribution `-0.005535`
- `lag_00__T1__duck_amount`: contribution `-0.002939`
- `lag_11__CT_place_TRUCK`: contribution `-0.002678`

Top utility-only movements:
- No utility movement among the top local contributors.
