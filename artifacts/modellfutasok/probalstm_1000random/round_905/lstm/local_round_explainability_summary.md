# Local Round Explainability

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `19`

## Largest probability jumps

- tick `126270`, seconds `40.50`, LSTM `0.1356`, delta `-0.2719`
- tick `126430`, seconds `43.00`, LSTM `0.0284`, delta `-0.0946`
- tick `125118`, seconds `22.50`, LSTM `0.3653`, delta `-0.0457`
- tick `126334`, seconds `41.50`, LSTM `0.1062`, delta `-0.0358`
- tick `124350`, seconds `10.50`, LSTM `0.2969`, delta `-0.0301`
- tick `125822`, seconds `33.50`, LSTM `0.3888`, delta `-0.0282`
- tick `124158`, seconds `7.50`, LSTM `0.3394`, delta `-0.0280`
- tick `126206`, seconds `39.50`, LSTM `0.4173`, delta `+0.0250`
- tick `126174`, seconds `39.00`, LSTM `0.3923`, delta `-0.0245`
- tick `124926`, seconds `19.50`, LSTM `0.3944`, delta `+0.0244`

## Top 15 local ridge features

- `lag_14__CT_place_EXTENDEDA`: coefficient `-0.002354`, |coef| `0.002354`
- `lag_04__T1__flash_duration`: coefficient `-0.001597`, |coef| `0.001597`
- `lag_06__T_place_MIDDOORS`: coefficient `-0.001517`, |coef| `0.001517`
- `lag_04__CT4__flash_duration`: coefficient `0.001491`, |coef| `0.001491`
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.001441`, |coef| `0.001441`
- `lag_09__CT_place_SHORTSTAIRS`: coefficient `-0.001439`, |coef| `0.001439`
- `lag_02__T_place_LOWERTUNNEL`: coefficient `0.001351`, |coef| `0.001351`
- `lag_07__T_place_LOWERTUNNEL`: coefficient `0.001269`, |coef| `0.001269`
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.001218`, |coef| `0.001218`
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.001083`, |coef| `0.001083`
- `lag_04__CT1__flash_duration`: coefficient `-0.001079`, |coef| `0.001079`
- `lag_00__T_kills_last_3s`: coefficient `-0.001047`, |coef| `0.001047`
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001022`, |coef| `0.001022`
- `lag_00__T_damage_last_5s`: coefficient `-0.000974`, |coef| `0.000974`
- `lag_07__CT4__shots_fired`: coefficient `0.000918`, |coef| `0.000918`

## Top 10 utility ridge features

- `lag_04__T1__flash_duration`: coefficient `-0.001597` (lowers CT win probability)
- `lag_04__CT4__flash_duration`: coefficient `0.001491` (raises CT win probability)
- `lag_11__CT_utility_damage_last_5s`: coefficient `-0.001218` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.001079` (lowers CT win probability)
- `lag_01__CT_utility_damage_last_5s`: coefficient `0.001022` (raises CT win probability)
- `lag_01__utility_damage_diff_last_5s`: coefficient `0.000873` (raises CT win probability)
- `lag_04__T_A_site_active_infernos`: coefficient `-0.000857` (lowers CT win probability)
- `lag_11__utility_damage_diff_last_5s`: coefficient `-0.000852` (lowers CT win probability)
- `lag_04__T_flash_duration_sum`: coefficient `-0.000689` (lowers CT win probability)
- `lag_09__CT4__flash_duration`: coefficient `0.000637` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_14__CT_place_EXTENDEDA`: coefficient `-0.002354` (lowers CT win probability)
- `lag_06__T_place_MIDDOORS`: coefficient `-0.001517` (lowers CT win probability)
- `lag_00__CT_place_SHORTSTAIRS`: coefficient `0.001441` (raises CT win probability)
- `lag_09__CT_place_SHORTSTAIRS`: coefficient `-0.001439` (lowers CT win probability)
- `lag_02__T_place_LOWERTUNNEL`: coefficient `0.001351` (raises CT win probability)
- `lag_07__T_place_LOWERTUNNEL`: coefficient `0.001269` (raises CT win probability)
- `lag_02__T_place_SHORTSTAIRS`: coefficient `-0.001083` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001047` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.000974` (lowers CT win probability)
- `lag_07__CT4__shots_fired`: coefficient `0.000918` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `126270`, seconds `40.50`, LSTM delta `-0.2719`

Top all feature movements:
- `lag_14__CT_place_EXTENDEDA`: contribution `-0.026429`
- `lag_04__CT4__flash_duration`: contribution `-0.010677`
- `lag_04__T1__flash_duration`: contribution `-0.010120`
- `lag_00__CT_place_SHORTSTAIRS`: contribution `-0.008032`
- `lag_09__CT_place_SHORTSTAIRS`: contribution `-0.008024`

Top utility-only movements:
- `lag_04__CT4__flash_duration`: contribution `-0.010677`
- `lag_04__T1__flash_duration`: contribution `-0.010120`
- `lag_04__CT1__flash_duration`: contribution `-0.006058`
- `lag_11__CT_utility_damage_last_5s`: contribution `-0.005633`
- `lag_01__CT_utility_damage_last_5s`: contribution `-0.004726`

### tick `126430`, seconds `43.00`, LSTM delta `-0.0946`

Top all feature movements:
- `lag_14__CT_place_EXTENDEDA`: contribution `+0.013214`
- `lag_07__T_place_LOWERTUNNEL`: contribution `-0.005488`
- `lag_03__T1__is_scoped`: contribution `-0.004673`
- `lag_09__CT4__flash_duration`: contribution `-0.004565`
- `lag_14__CT_place_SHORTSTAIRS`: contribution `-0.003622`

Top utility-only movements:
- `lag_09__CT4__flash_duration`: contribution `-0.004565`
- `lag_09__T1__flash_duration`: contribution `-0.003466`

### tick `125118`, seconds `22.50`, LSTM delta `-0.0457`

Top all feature movements:
- `lag_07__T_place_LOWERTUNNEL`: contribution `+0.005488`
- `lag_07__T_place_TUNNELSTAIRS`: contribution `-0.004383`
- `lag_09__T_place_TUNNELSTAIRS`: contribution `-0.002423`
- `lag_01__CT1__duck_amount`: contribution `-0.002314`
- `lag_13__T_place_TUNNELSTAIRS`: contribution `-0.002246`

Top utility-only movements:
- `lag_06__CT5__flash_duration`: contribution `-0.001715`
- `lag_06__CT_flash_duration_sum`: contribution `-0.001300`

### tick `126334`, seconds `41.50`, LSTM delta `-0.0358`

Top all feature movements:
- `lag_06__T1__flash_duration`: contribution `-0.003422`
- `lag_05__T5__duck_amount`: contribution `+0.002576`
- `lag_08__T_place_MIDDOORS`: contribution `-0.002421`
- `lag_04__T_place_LOWERTUNNEL`: contribution `-0.002348`
- `lag_00__T1__is_scoped`: contribution `-0.002213`

Top utility-only movements:
- `lag_06__T1__flash_duration`: contribution `-0.003422`
- `lag_04__CT1__flash_duration`: contribution `+0.002197`
- `lag_03__CT_utility_damage_last_5s`: contribution `-0.001449`
- `lag_03__utility_damage_diff_last_5s`: contribution `-0.001446`
- `lag_13__CT_utility_damage_last_5s`: contribution `-0.001391`

### tick `124350`, seconds `10.50`, LSTM delta `-0.0301`

Top all feature movements:
- `lag_10__T_he_last_5s`: contribution `-0.005601`
- `lag_14__CT_place_MIDDOORS`: contribution `-0.003306`
- `lag_14__T_place_OUTSIDETUNNEL`: contribution `-0.002309`
- `lag_12__CT_place_UNDERA`: contribution `+0.002155`
- `lag_03__CT5__is_walking`: contribution `-0.002080`

Top utility-only movements:
- `lag_10__T_he_last_5s`: contribution `-0.005601`
