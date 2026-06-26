# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `19`

## Largest probability jumps

- tick `196039`, seconds `81.50`, LSTM `0.1637`, delta `-0.3104`
- tick `196743`, seconds `92.50`, LSTM `0.1361`, delta `-0.2303`
- tick `196615`, seconds `90.50`, LSTM `0.3857`, delta `+0.1831`
- tick `192551`, seconds `27.00`, LSTM `0.3757`, delta `+0.0835`
- tick `191943`, seconds `17.50`, LSTM `0.2792`, delta `-0.0782`
- tick `192871`, seconds `32.00`, LSTM `0.2757`, delta `-0.0734`
- tick `190855`, seconds `0.50`, LSTM `0.2418`, delta `-0.0635`
- tick `195271`, seconds `69.50`, LSTM `0.3132`, delta `+0.0461`
- tick `194023`, seconds `50.00`, LSTM `0.2973`, delta `+0.0440`
- tick `195623`, seconds `75.00`, LSTM `0.4710`, delta `+0.0399`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002549`, |coef| `0.002549`
- `lag_08__CT_place_TOPOFMID`: coefficient `-0.002173`, |coef| `0.002173`
- `lag_07__CT_place_TOPOFMID`: coefficient `-0.002163`, |coef| `0.002163`
- `lag_13__CT_place_PIT`: coefficient `-0.002135`, |coef| `0.002135`
- `lag_00__T_kills_last_3s`: coefficient `-0.002106`, |coef| `0.002106`
- `lag_05__CT_place_LIBRARY`: coefficient `0.001931`, |coef| `0.001931`
- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001918`, |coef| `0.001918`
- `lag_00__T_shots_fired_sum`: coefficient `-0.001917`, |coef| `0.001917`
- `lag_00__damage_diff_last_5s`: coefficient `0.001830`, |coef| `0.001830`
- `lag_07__CT_place_PIT`: coefficient `0.001797`, |coef| `0.001797`
- `lag_09__CT_flashed_players`: coefficient `-0.001782`, |coef| `0.001782`
- `lag_00__CT_place_TOPOFMID`: coefficient `0.001678`, |coef| `0.001678`
- `lag_05__CT2__flash_duration`: coefficient `-0.001664`, |coef| `0.001664`
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.001662`, |coef| `0.001662`
- `lag_01__T2__flash_duration`: coefficient `0.001576`, |coef| `0.001576`

## Top 10 utility ridge features

- `lag_14__CT_utility_damage_last_5s`: coefficient `0.001918` (raises CT win probability)
- `lag_05__CT2__flash_duration`: coefficient `-0.001664` (lowers CT win probability)
- `lag_02__CT_B_site_active_infernos`: coefficient `-0.001662` (lowers CT win probability)
- `lag_01__T2__flash_duration`: coefficient `0.001576` (raises CT win probability)
- `lag_05__T5__flash_duration`: coefficient `-0.001451` (lowers CT win probability)
- `lag_05__T2__flash_duration`: coefficient `-0.001404` (lowers CT win probability)
- `lag_00__utility_damage_diff_last_5s`: coefficient `0.001362` (raises CT win probability)
- `lag_01__T5__flash_duration`: coefficient `0.001356` (raises CT win probability)
- `lag_01__CT2__flash_duration`: coefficient `0.001321` (raises CT win probability)
- `lag_14__utility_damage_diff_last_5s`: coefficient `0.001318` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.002549` (raises CT win probability)
- `lag_08__CT_place_TOPOFMID`: coefficient `-0.002173` (lowers CT win probability)
- `lag_07__CT_place_TOPOFMID`: coefficient `-0.002163` (lowers CT win probability)
- `lag_13__CT_place_PIT`: coefficient `-0.002135` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002106` (lowers CT win probability)
- `lag_05__CT_place_LIBRARY`: coefficient `0.001931` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.001917` (lowers CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001830` (raises CT win probability)
- `lag_07__CT_place_PIT`: coefficient `0.001797` (raises CT win probability)
- `lag_09__CT_flashed_players`: coefficient `-0.001782` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `196039`, seconds `81.50`, LSTM delta `-0.3104`

Top all feature movements:
- `lag_05__CT_place_LIBRARY`: contribution `-0.012382`
- `lag_13__CT_place_PIT`: contribution `-0.009192`
- `lag_06__CT_place_LIBRARY`: contribution `-0.008804`
- `lag_08__CT_place_TOPOFMID`: contribution `-0.007886`
- `lag_07__CT_place_TOPOFMID`: contribution `-0.007850`

Top utility-only movements:
- `lag_14__CT_utility_damage_last_5s`: contribution `-0.007598`
- `lag_02__CT_B_site_active_infernos`: contribution `-0.005710`
- `lag_14__utility_damage_diff_last_5s`: contribution `-0.004285`

### tick `196743`, seconds `92.50`, LSTM delta `-0.2303`

Top all feature movements:
- `lag_05__CT2__flash_duration`: contribution `-0.010492`
- `lag_00__CT_place_LIBRARY`: contribution `-0.009800`
- `lag_05__T2__flash_duration`: contribution `-0.008646`
- `lag_05__T5__flash_duration`: contribution `-0.008390`
- `lag_00__T_kills_last_3s`: contribution `-0.006671`

Top utility-only movements:
- `lag_05__CT2__flash_duration`: contribution `-0.010492`
- `lag_05__T2__flash_duration`: contribution `-0.008646`
- `lag_05__T5__flash_duration`: contribution `-0.008390`
- `lag_04__T5__flash_duration`: contribution `-0.005948`
- `lag_05__T_flash_duration_sum`: contribution `-0.004988`

### tick `196615`, seconds `90.50`, LSTM delta `+0.1831`

Top all feature movements:
- `lag_01__T2__flash_duration`: contribution `+0.009705`
- `lag_01__CT2__flash_duration`: contribution `+0.008328`
- `lag_01__T5__flash_duration`: contribution `+0.007836`
- `lag_00__T5__flash_duration`: contribution `+0.007098`
- `lag_01__CT_place_ARCH`: contribution `+0.006327`

Top utility-only movements:
- `lag_01__T2__flash_duration`: contribution `+0.009705`
- `lag_01__CT2__flash_duration`: contribution `+0.008328`
- `lag_01__T5__flash_duration`: contribution `+0.007836`
- `lag_00__T5__flash_duration`: contribution `+0.007098`
- `lag_01__T_flash_duration_sum`: contribution `+0.005053`

### tick `192551`, seconds `27.00`, LSTM delta `+0.0835`

Top all feature movements:
- `lag_12__CT_flashes_last_5s`: contribution `+0.010109`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007871`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007745`
- `lag_12__CT3__flash`: contribution `+0.004084`
- `lag_07__CT4__is_walking`: contribution `+0.003365`

Top utility-only movements:
- `lag_12__CT_flashes_last_5s`: contribution `+0.010109`
- `lag_00__CT_utility_damage_last_5s`: contribution `+0.007871`
- `lag_00__utility_damage_diff_last_5s`: contribution `+0.007745`
- `lag_12__CT3__flash`: contribution `+0.004084`
- `lag_09__T_utility_damage_last_5s`: contribution `+0.002299`

### tick `191943`, seconds `17.50`, LSTM delta `-0.0782`

Top all feature movements:
- `lag_05__CT_place_BALCONY`: contribution `-0.005773`
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.005655`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004471`
- `lag_04__CT1__flash_duration`: contribution `-0.004141`
- `lag_12__CT_place_TOPOFMID`: contribution `-0.002926`

Top utility-only movements:
- `lag_00__utility_damage_diff_last_5s`: contribution `-0.005655`
- `lag_00__T_utility_damage_last_5s`: contribution `-0.004471`
- `lag_04__CT1__flash_duration`: contribution `-0.004141`
