# Local Round Explainability

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `3`

## Largest probability jumps

- tick `21289`, seconds `94.50`, LSTM `0.2195`, delta `-0.3376`
- tick `23113`, seconds `123.00`, LSTM `0.2928`, delta `-0.2181`
- tick `22601`, seconds `115.00`, LSTM `0.3368`, delta `+0.1927`
- tick `21129`, seconds `92.00`, LSTM `0.2793`, delta `-0.1922`
- tick `21193`, seconds `93.00`, LSTM `0.5632`, delta `+0.1820`
- tick `23177`, seconds `124.00`, LSTM `0.0668`, delta `-0.1794`
- tick `21161`, seconds `92.50`, LSTM `0.3812`, delta `+0.1019`
- tick `22953`, seconds `120.50`, LSTM `0.4808`, delta `+0.0608`
- tick `21321`, seconds `95.00`, LSTM `0.2701`, delta `+0.0506`
- tick `23145`, seconds `123.50`, LSTM `0.2463`, delta `-0.0465`

## Top 15 local ridge features

- `lag_05__CT_defusing_count`: coefficient `-0.005709`, |coef| `0.005709`
- `lag_00__kill_diff_last_3s`: coefficient `0.004700`, |coef| `0.004700`
- `lag_00__T_kills_last_3s`: coefficient `-0.003668`, |coef| `0.003668`
- `lag_07__CT_defusing_count`: coefficient `-0.003466`, |coef| `0.003466`
- `lag_00__damage_diff_last_5s`: coefficient `0.003408`, |coef| `0.003408`
- `lag_15__T_place_UPPERPARK`: coefficient `0.003142`, |coef| `0.003142`
- `lag_12__CT_place_BACKOFA`: coefficient `0.002934`, |coef| `0.002934`
- `lag_05__CT_place_STORAGEROOM`: coefficient `-0.002322`, |coef| `0.002322`
- `lag_14__T_place_UPPERPARK`: coefficient `0.002319`, |coef| `0.002319`
- `lag_00__T_damage_last_5s`: coefficient `-0.002309`, |coef| `0.002309`
- `lag_00__CT_kills_last_3s`: coefficient `0.002295`, |coef| `0.002295`
- `lag_00__CT_place_STORAGEROOM`: coefficient `-0.002175`, |coef| `0.002175`
- `lag_02__CT3__duck_amount`: coefficient `0.002099`, |coef| `0.002099`
- `lag_06__CT_defusing_count`: coefficient `-0.002052`, |coef| `0.002052`
- `lag_06__CT3__duck_amount`: coefficient `0.002030`, |coef| `0.002030`

## Top 10 utility ridge features

- `lag_03__CT1__flash_duration`: coefficient `-0.001560` (lowers CT win probability)
- `lag_13__T_A_site_active_smokes`: coefficient `-0.001476` (lowers CT win probability)
- `lag_10__T_A_site_active_smokes`: coefficient `-0.001338` (lowers CT win probability)
- `lag_09__CT1__flash_duration`: coefficient `-0.001146` (lowers CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `-0.001138` (lowers CT win probability)
- `lag_09__T_A_site_active_smokes`: coefficient `-0.001108` (lowers CT win probability)
- `lag_14__T2__flash_duration`: coefficient `0.001096` (raises CT win probability)
- `lag_13__T_active_smokes`: coefficient `-0.001071` (lowers CT win probability)
- `lag_11__T_A_site_active_smokes`: coefficient `-0.001067` (lowers CT win probability)
- `lag_14__CT1__flash_duration`: coefficient `-0.000993` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_05__CT_defusing_count`: coefficient `-0.005709` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.004700` (raises CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.003668` (lowers CT win probability)
- `lag_07__CT_defusing_count`: coefficient `-0.003466` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.003408` (raises CT win probability)
- `lag_15__T_place_UPPERPARK`: coefficient `0.003142` (raises CT win probability)
- `lag_12__CT_place_BACKOFA`: coefficient `0.002934` (raises CT win probability)
- `lag_05__CT_place_STORAGEROOM`: coefficient `-0.002322` (lowers CT win probability)
- `lag_14__T_place_UPPERPARK`: coefficient `0.002319` (raises CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.002309` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `21289`, seconds `94.50`, LSTM delta `-0.3376`

Top all feature movements:
- `lag_05__CT_place_STORAGEROOM`: contribution `-0.049667`
- `lag_02__CT_place_BACKOFA`: contribution `-0.013860`
- `lag_11__CT_place_STAIRS`: contribution `-0.013189`
- `lag_08__CT_place_STAIRS`: contribution `-0.012956`
- `lag_00__T_kills_last_3s`: contribution `-0.011620`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `-0.010468`
- `lag_14__CT1__flash_duration`: contribution `-0.005721`
- `lag_15__T4__flash_duration`: contribution `-0.004629`
- `lag_08__CT4__flash_duration`: contribution `-0.004312`

### tick `23113`, seconds `123.00`, LSTM delta `-0.2181`

Top all feature movements:
- `lag_05__CT_defusing_count`: contribution `-0.055347`
- `lag_12__CT_place_BACKOFA`: contribution `-0.028335`
- `lag_00__T_kills_last_3s`: contribution `-0.011620`
- `lag_00__kill_diff_last_3s`: contribution `-0.011313`
- `lag_00__CT_place_LOWERPARK`: contribution `-0.008722`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `22601`, seconds `115.00`, LSTM delta `+0.1927`

Top all feature movements:
- `lag_15__T_place_UPPERPARK`: contribution `+0.016569`
- `lag_00__kill_diff_last_3s`: contribution `+0.011313`
- `lag_02__CT3__duck_amount`: contribution `+0.007406`
- `lag_00__CT_kills_last_3s`: contribution `+0.006626`
- `lag_15__CT3__duck_amount`: contribution `+0.005575`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `21129`, seconds `92.00`, LSTM delta `-0.1922`

Top all feature movements:
- `lag_00__CT_place_STORAGEROOM`: contribution `-0.046527`
- `lag_07__CT_place_CONSTRUCTION`: contribution `-0.014746`
- `lag_00__T_kills_last_3s`: contribution `-0.011620`
- `lag_00__kill_diff_last_3s`: contribution `-0.011313`
- `lag_03__CT_place_BACKOFA`: contribution `-0.010007`

Top utility-only movements:
- `lag_09__CT1__flash_duration`: contribution `-0.006601`
- `lag_03__CT4__flash_duration`: contribution `-0.004119`
- `lag_10__T4__flash_duration`: contribution `-0.003790`
- `lag_13__T2__flash_duration`: contribution `-0.002775`

### tick `21193`, seconds `93.00`, LSTM delta `+0.1820`

Top all feature movements:
- `lag_02__CT_place_STORAGEROOM`: contribution `+0.028477`
- `lag_02__CT_place_BACKOFA`: contribution `+0.013860`
- `lag_08__CT_place_STAIRS`: contribution `+0.012956`
- `lag_00__kill_diff_last_3s`: contribution `+0.011313`
- `lag_09__CT_place_CONSTRUCTION`: contribution `+0.010939`

Top utility-only movements:
- `lag_03__CT1__flash_duration`: contribution `+0.007122`
- `lag_12__T4__flash_duration`: contribution `+0.003556`
