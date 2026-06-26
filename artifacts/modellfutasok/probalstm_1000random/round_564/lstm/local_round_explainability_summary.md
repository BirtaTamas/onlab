# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-spirit-bo3-NmwBJVzYbgyZgcQrbNESHr/flyquest-vs-spirit-m1-anubis.csv`
- round_num: `14`

## Largest probability jumps

- tick `121815`, seconds `60.50`, LSTM `0.0395`, delta `-0.1571`
- tick `121623`, seconds `57.50`, LSTM `0.1743`, delta `+0.1288`
- tick `121655`, seconds `58.00`, LSTM `0.1013`, delta `-0.0730`
- tick `121783`, seconds `60.00`, LSTM `0.1966`, delta `+0.0663`
- tick `121719`, seconds `59.00`, LSTM `0.1432`, delta `+0.0506`
- tick `121847`, seconds `61.00`, LSTM `0.0881`, delta `+0.0485`
- tick `117975`, seconds `0.50`, LSTM `0.0225`, delta `-0.0319`
- tick `121591`, seconds `57.00`, LSTM `0.0455`, delta `+0.0263`
- tick `121879`, seconds `61.50`, LSTM `0.0675`, delta `-0.0206`
- tick `121911`, seconds `62.00`, LSTM `0.0473`, delta `-0.0202`

## Top 15 local ridge features

- `lag_04__T2__flash_duration`: coefficient `0.001197`, |coef| `0.001197`
- `lag_04__T4__flash_duration`: coefficient `0.001166`, |coef| `0.001166`
- `lag_00__kill_diff_last_3s`: coefficient `0.001087`, |coef| `0.001087`
- `lag_04__T_flash_duration_sum`: coefficient `0.000965`, |coef| `0.000965`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.000956`, |coef| `0.000956`
- `lag_02__T4__flash_duration`: coefficient `-0.000932`, |coef| `0.000932`
- `lag_00__T2__flash_duration`: coefficient `-0.000895`, |coef| `0.000895`
- `lag_03__T4__flash_duration`: coefficient `0.000887`, |coef| `0.000887`
- `lag_04__CT3__flash_duration`: coefficient `0.000864`, |coef| `0.000864`
- `lag_12__T_place_MAIN`: coefficient `-0.000846`, |coef| `0.000846`
- `lag_02__CT_place_OUTSIDELONG`: coefficient `-0.000808`, |coef| `0.000808`
- `lag_03__T3__duck_amount`: coefficient `0.000796`, |coef| `0.000796`
- `lag_05__T4__flash_duration`: coefficient `-0.000766`, |coef| `0.000766`
- `lag_00__CT_kills_last_3s`: coefficient `0.000757`, |coef| `0.000757`
- `lag_05__T2__flash_duration`: coefficient `-0.000740`, |coef| `0.000740`

## Top 10 utility ridge features

- `lag_04__T2__flash_duration`: coefficient `0.001197` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.001166` (raises CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `0.000965` (raises CT win probability)
- `lag_02__T4__flash_duration`: coefficient `-0.000932` (lowers CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000895` (lowers CT win probability)
- `lag_03__T4__flash_duration`: coefficient `0.000887` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `0.000864` (raises CT win probability)
- `lag_05__T4__flash_duration`: coefficient `-0.000766` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.000740` (lowers CT win probability)
- `lag_06__T2__flash_duration`: coefficient `0.000712` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.001087` (raises CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.000956` (raises CT win probability)
- `lag_12__T_place_MAIN`: coefficient `-0.000846` (lowers CT win probability)
- `lag_02__CT_place_OUTSIDELONG`: coefficient `-0.000808` (lowers CT win probability)
- `lag_03__T3__duck_amount`: coefficient `0.000796` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000757` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000722` (lowers CT win probability)
- `lag_09__CT2__duck_amount`: coefficient `0.000715` (raises CT win probability)
- `lag_06__CT1__duck_amount`: coefficient `0.000706` (raises CT win probability)
- `lag_01__CT1__duck_amount`: coefficient `0.000694` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `121815`, seconds `60.50`, LSTM delta `-0.1571`

Top all feature movements:
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.009702`
- `lag_02__CT_place_OUTSIDELONG`: contribution `-0.008200`
- `lag_00__kill_diff_last_3s`: contribution `-0.007851`
- `lag_00__T_shots_fired_sum`: contribution `-0.006495`
- `lag_06__T2__flash_duration`: contribution `-0.005141`

Top utility-only movements:
- `lag_06__T2__flash_duration`: contribution `-0.005141`
- `lag_08__T4__flash_duration`: contribution `-0.004919`
- `lag_10__T2__flash_duration`: contribution `-0.004734`
- `lag_10__T4__flash_duration`: contribution `-0.004149`
- `lag_02__CT3__flash_duration`: contribution `-0.003001`

### tick `121623`, seconds `57.50`, LSTM delta `+0.1288`

Top all feature movements:
- `lag_04__T2__flash_duration`: contribution `+0.008647`
- `lag_04__T4__flash_duration`: contribution `+0.008416`
- `lag_02__T4__flash_duration`: contribution `+0.006722`
- `lag_00__T2__flash_duration`: contribution `+0.006462`
- `lag_04__T_flash_duration_sum`: contribution `+0.005684`

Top utility-only movements:
- `lag_04__T2__flash_duration`: contribution `+0.008647`
- `lag_04__T4__flash_duration`: contribution `+0.008416`
- `lag_02__T4__flash_duration`: contribution `+0.006722`
- `lag_00__T2__flash_duration`: contribution `+0.006462`
- `lag_04__T_flash_duration_sum`: contribution `+0.005684`

### tick `121655`, seconds `58.00`, LSTM delta `-0.0730`

Top all feature movements:
- `lag_03__T4__flash_duration`: contribution `-0.006400`
- `lag_05__T4__flash_duration`: contribution `-0.005530`
- `lag_05__T2__flash_duration`: contribution `-0.005341`
- `lag_01__T2__flash_duration`: contribution `-0.003694`
- `lag_05__T_flash_duration_sum`: contribution `-0.003615`

Top utility-only movements:
- `lag_03__T4__flash_duration`: contribution `-0.006400`
- `lag_05__T4__flash_duration`: contribution `-0.005530`
- `lag_05__T2__flash_duration`: contribution `-0.005341`
- `lag_01__T2__flash_duration`: contribution `-0.003694`
- `lag_05__T_flash_duration_sum`: contribution `-0.003615`

### tick `121783`, seconds `60.00`, LSTM delta `+0.0663`

Top all feature movements:
- `lag_05__T2__flash_duration`: contribution `+0.005341`
- `lag_09__T2__flash_duration`: contribution `+0.003866`
- `lag_01__CT_place_OUTSIDELONG`: contribution `+0.003716`
- `lag_04__CT_place_CONNECTOR`: contribution `+0.002823`
- `lag_06__CT1__duck_amount`: contribution `+0.002694`

Top utility-only movements:
- `lag_05__T2__flash_duration`: contribution `+0.005341`
- `lag_09__T2__flash_duration`: contribution `+0.003866`
- `lag_09__T_flash_duration_sum`: contribution `+0.001915`
- `lag_09__T4__flash_duration`: contribution `+0.001874`
- `lag_05__T_flash_duration_sum`: contribution `+0.001814`

### tick `121719`, seconds `59.00`, LSTM delta `+0.0506`

Top all feature movements:
- `lag_05__T4__flash_duration`: contribution `+0.005530`
- `lag_02__CT_place_CONNECTOR`: contribution `+0.003259`
- `lag_03__T3__duck_amount`: contribution `+0.003001`
- `lag_02__CT_place_CANAL`: contribution `+0.002762`
- `lag_04__CT2__duck_amount`: contribution `+0.002362`

Top utility-only movements:
- `lag_05__T4__flash_duration`: contribution `+0.005530`
- `lag_07__T2__flash_duration`: contribution `+0.002073`
- `lag_05__T_flash_duration_sum`: contribution `+0.001801`
- `lag_07__CT3__flash_duration`: contribution `+0.001525`
- `lag_03__T_flash_duration_sum`: contribution `-0.001211`
