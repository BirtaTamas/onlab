# Local Round Explainability

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `8`

## Largest probability jumps

- tick `64316`, seconds `50.50`, LSTM `0.7511`, delta `+0.2022`
- tick `64348`, seconds `51.00`, LSTM `0.9033`, delta `+0.1522`
- tick `63164`, seconds `32.50`, LSTM `0.6779`, delta `+0.1337`
- tick `63708`, seconds `41.00`, LSTM `0.5219`, delta `-0.1188`
- tick `64444`, seconds `52.50`, LSTM `0.9507`, delta `+0.1122`
- tick `63868`, seconds `43.50`, LSTM `0.3993`, delta `-0.0853`
- tick `64284`, seconds `50.00`, LSTM `0.5489`, delta `+0.0798`
- tick `64380`, seconds `51.50`, LSTM `0.8275`, delta `-0.0757`
- tick `64220`, seconds `49.00`, LSTM `0.4239`, delta `-0.0618`
- tick `64028`, seconds `46.00`, LSTM `0.5086`, delta `+0.0592`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002402`, |coef| `0.002402`
- `lag_00__CT_kills_last_3s`: coefficient `0.002100`, |coef| `0.002100`
- `lag_00__T4__flash_duration`: coefficient `0.002076`, |coef| `0.002076`
- `lag_01__T3__flash_duration`: coefficient `-0.002024`, |coef| `0.002024`
- `lag_00__CT_shots_fired_sum`: coefficient `0.001916`, |coef| `0.001916`
- `lag_00__CT3__is_scoped`: coefficient `0.001836`, |coef| `0.001836`
- `lag_00__damage_diff_last_5s`: coefficient `0.001743`, |coef| `0.001743`
- `lag_01__T4__flash_duration`: coefficient `0.001618`, |coef| `0.001618`
- `lag_00__CT_damage_last_5s`: coefficient `0.001406`, |coef| `0.001406`
- `lag_02__T3__flash_duration`: coefficient `-0.001354`, |coef| `0.001354`
- `lag_09__CT3__is_scoped`: coefficient `0.001354`, |coef| `0.001354`
- `lag_00__T3__flash_duration`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_07__CT3__is_scoped`: coefficient `0.001327`, |coef| `0.001327`
- `lag_00__CT4__flash_duration`: coefficient `0.001277`, |coef| `0.001277`
- `lag_04__CT3__flash_duration`: coefficient `-0.001142`, |coef| `0.001142`

## Top 10 utility ridge features

- `lag_00__T4__flash_duration`: coefficient `0.002076` (raises CT win probability)
- `lag_01__T3__flash_duration`: coefficient `-0.002024` (lowers CT win probability)
- `lag_01__T4__flash_duration`: coefficient `0.001618` (raises CT win probability)
- `lag_02__T3__flash_duration`: coefficient `-0.001354` (lowers CT win probability)
- `lag_00__T3__flash_duration`: coefficient `-0.001342` (lowers CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.001277` (raises CT win probability)
- `lag_04__CT3__flash_duration`: coefficient `-0.001142` (lowers CT win probability)
- `lag_15__CT5__flash_duration`: coefficient `0.001036` (raises CT win probability)
- `lag_04__T4__flash_duration`: coefficient `0.001025` (raises CT win probability)
- `lag_14__T3__flash_duration`: coefficient `0.001012` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002402` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.002100` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.001916` (raises CT win probability)
- `lag_00__CT3__is_scoped`: coefficient `0.001836` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001743` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001406` (raises CT win probability)
- `lag_09__CT3__is_scoped`: coefficient `0.001354` (raises CT win probability)
- `lag_07__CT3__is_scoped`: coefficient `0.001327` (raises CT win probability)
- `lag_15__T_place_TMAIN`: coefficient `-0.001112` (lowers CT win probability)
- `lag_08__CT3__is_scoped`: coefficient `0.001104` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `64316`, seconds `50.50`, LSTM delta `+0.2022`

Top all feature movements:
- `lag_01__T3__flash_duration`: contribution `+0.015023`
- `lag_00__T4__flash_duration`: contribution `+0.013629`
- `lag_00__CT3__is_scoped`: contribution `+0.008351`
- `lag_04__CT3__flash_duration`: contribution `+0.006891`
- `lag_00__CT4__flash_duration`: contribution `+0.006160`

Top utility-only movements:
- `lag_01__T3__flash_duration`: contribution `+0.015023`
- `lag_00__T4__flash_duration`: contribution `+0.013629`
- `lag_04__CT3__flash_duration`: contribution `+0.006891`
- `lag_00__CT4__flash_duration`: contribution `+0.006160`
- `lag_05__T1__flash_duration`: contribution `+0.004639`

### tick `64348`, seconds `51.00`, LSTM delta `+0.1522`

Top all feature movements:
- `lag_01__T4__flash_duration`: contribution `+0.010619`
- `lag_02__T3__flash_duration`: contribution `+0.010050`
- `lag_00__CT_kills_last_3s`: contribution `+0.006064`
- `lag_00__kill_diff_last_3s`: contribution `+0.005782`
- `lag_08__CT3__is_scoped`: contribution `+0.005023`

Top utility-only movements:
- `lag_01__T4__flash_duration`: contribution `+0.010619`
- `lag_02__T3__flash_duration`: contribution `+0.010050`
- `lag_01__CT4__flash_duration`: contribution `+0.004592`
- `lag_05__CT3__flash_duration`: contribution `+0.004045`
- `lag_08__CT2__flash_duration`: contribution `+0.003952`

### tick `63164`, seconds `32.50`, LSTM delta `+0.1337`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006657`
- `lag_09__CT3__is_scoped`: contribution `+0.006157`
- `lag_00__CT_kills_last_3s`: contribution `+0.006064`
- `lag_00__kill_diff_last_3s`: contribution `+0.005782`
- `lag_10__T3__shots_fired`: contribution `+0.004969`

Top utility-only movements:
- `lag_09__CT_A_site_active_infernos`: contribution `+0.002562`
- `lag_00__T2__flash`: contribution `+0.001909`

### tick `63708`, seconds `41.00`, LSTM delta `-0.1188`

Top all feature movements:
- `lag_11__CT_shots_fired_sum`: contribution `-0.011749`
- `lag_09__T_place_DUMPSTER`: contribution `-0.009566`
- `lag_13__T_place_DUMPSTER`: contribution `-0.008957`
- `lag_11__CT5__shots_fired`: contribution `-0.006817`
- `lag_00__kill_diff_last_3s`: contribution `-0.005782`

Top utility-only movements:
- `lag_07__CT_active_infernos`: contribution `-0.003396`
- `lag_07__CT_A_site_active_infernos`: contribution `-0.002163`
- `lag_07__CT_B_site_active_infernos`: contribution `-0.001715`

### tick `64444`, seconds `52.50`, LSTM delta `+0.1122`

Top all feature movements:
- `lag_04__T4__flash_duration`: contribution `+0.006727`
- `lag_00__CT_shots_fired_sum`: contribution `+0.006657`
- `lag_00__CT_kills_last_3s`: contribution `+0.006064`
- `lag_07__CT3__is_scoped`: contribution `-0.006035`
- `lag_00__kill_diff_last_3s`: contribution `+0.005782`

Top utility-only movements:
- `lag_04__T4__flash_duration`: contribution `+0.006727`
- `lag_08__CT3__flash_duration`: contribution `+0.003199`
- `lag_11__CT2__flash_duration`: contribution `+0.002741`
- `lag_09__T1__flash_duration`: contribution `+0.002326`
- `lag_04__CT4__flash_duration`: contribution `+0.002210`
