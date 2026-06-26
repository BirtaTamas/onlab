# Local Round Explainability

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `13`

## Largest probability jumps

- tick `115536`, seconds `20.50`, LSTM `0.8734`, delta `+0.1584`
- tick `115408`, seconds `18.50`, LSTM `0.6597`, delta `+0.1455`
- tick `120400`, seconds `96.50`, LSTM `0.8196`, delta `-0.1085`
- tick `120560`, seconds `99.00`, LSTM `0.8234`, delta `-0.1076`
- tick `120432`, seconds `97.00`, LSTM `0.9246`, delta `+0.1050`
- tick `115568`, seconds `21.00`, LSTM `0.9327`, delta `+0.0593`
- tick `115760`, seconds `24.00`, LSTM `0.9089`, delta `-0.0544`
- tick `121040`, seconds `106.50`, LSTM `0.7655`, delta `+0.0525`
- tick `121072`, seconds `107.00`, LSTM `0.8166`, delta `+0.0511`
- tick `115504`, seconds `20.00`, LSTM `0.7150`, delta `+0.0345`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003868`, |coef| `0.003868`
- `lag_00__damage_diff_last_5s`: coefficient `0.003276`, |coef| `0.003276`
- `lag_00__CT_kills_last_3s`: coefficient `0.002880`, |coef| `0.002880`
- `lag_11__T_place_UNDERPASS`: coefficient `0.002108`, |coef| `0.002108`
- `lag_00__T4__is_walking`: coefficient `-0.002107`, |coef| `0.002107`
- `lag_07__CT2__duck_amount`: coefficient `-0.002026`, |coef| `0.002026`
- `lag_00__T_kills_last_3s`: coefficient `-0.001931`, |coef| `0.001931`
- `lag_00__CT_damage_last_5s`: coefficient `0.001923`, |coef| `0.001923`
- `lag_13__T_place_UNDERPASS`: coefficient `0.001909`, |coef| `0.001909`
- `lag_15__CT3__is_walking`: coefficient `0.001849`, |coef| `0.001849`
- `lag_13__CT_place_APARTMENTS`: coefficient `0.001849`, |coef| `0.001849`
- `lag_01__CT_place_SHOP`: coefficient `-0.001707`, |coef| `0.001707`
- `lag_04__T_place_UNDERPASS`: coefficient `-0.001697`, |coef| `0.001697`
- `lag_00__CT4__duck_amount`: coefficient `0.001629`, |coef| `0.001629`
- `lag_00__alive_diff`: coefficient `0.001626`, |coef| `0.001626`

## Top 10 utility ridge features

- `lag_07__T1__smoke`: coefficient `-0.001290` (lowers CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `-0.000971` (lowers CT win probability)
- `lag_03__T_B_site_active_infernos`: coefficient `0.000961` (raises CT win probability)
- `lag_15__T4__molly`: coefficient `0.000933` (raises CT win probability)
- `lag_03__T1__smoke`: coefficient `-0.000932` (lowers CT win probability)
- `lag_07__T2__smoke`: coefficient `0.000782` (raises CT win probability)
- `lag_12__T_active_infernos`: coefficient `-0.000738` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `0.000693` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.000651` (raises CT win probability)
- `lag_11__T_B_site_active_infernos`: coefficient `-0.000634` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.003868` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003276` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002880` (raises CT win probability)
- `lag_11__T_place_UNDERPASS`: coefficient `0.002108` (raises CT win probability)
- `lag_00__T4__is_walking`: coefficient `-0.002107` (lowers CT win probability)
- `lag_07__CT2__duck_amount`: coefficient `-0.002026` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001931` (lowers CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001923` (raises CT win probability)
- `lag_13__T_place_UNDERPASS`: coefficient `0.001909` (raises CT win probability)
- `lag_15__CT3__is_walking`: coefficient `0.001849` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `115536`, seconds `20.50`, LSTM delta `+0.1584`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009309`
- `lag_00__CT_kills_last_3s`: contribution `+0.008315`
- `lag_11__T_place_UNDERPASS`: contribution `+0.008256`
- `lag_13__T_place_UNDERPASS`: contribution `+0.007476`
- `lag_00__damage_diff_last_5s`: contribution `+0.007389`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `115408`, seconds `18.50`, LSTM delta `+0.1455`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009309`
- `lag_00__CT_kills_last_3s`: contribution `+0.008315`
- `lag_00__damage_diff_last_5s`: contribution `+0.007389`
- `lag_13__CT_place_APARTMENTS`: contribution `+0.007104`
- `lag_02__T_place_HOUSE`: contribution `+0.006758`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `120400`, seconds `96.50`, LSTM delta `-0.1085`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009309`
- `lag_07__CT2__duck_amount`: contribution `-0.007719`
- `lag_00__T_kills_last_3s`: contribution `-0.006117`
- `lag_04__CT2__duck_amount`: contribution `-0.005448`
- `lag_04__T4__duck_amount`: contribution `-0.004937`

Top utility-only movements:
- `lag_12__T_B_site_active_infernos`: contribution `-0.002744`
- `lag_15__T4__molly`: contribution `-0.002033`

### tick `120560`, seconds `99.00`, LSTM delta `-0.1076`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.009309`
- `lag_01__CT_place_SHOP`: contribution `-0.008563`
- `lag_00__damage_diff_last_5s`: contribution `-0.007463`
- `lag_00__T_kills_last_3s`: contribution `-0.006117`
- `lag_12__CT2__duck_amount`: contribution `-0.005251`

Top utility-only movements:
- `lag_03__T_B_site_active_infernos`: contribution `-0.002716`

### tick `120432`, seconds `97.00`, LSTM delta `+0.1050`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.009309`
- `lag_01__CT_place_SHOP`: contribution `+0.008563`
- `lag_00__CT_kills_last_3s`: contribution `+0.008315`
- `lag_07__CT2__duck_amount`: contribution `+0.007719`
- `lag_04__CT2__duck_amount`: contribution `+0.005448`

Top utility-only movements:
- `lag_13__T_B_site_active_infernos`: contribution `+0.001841`
