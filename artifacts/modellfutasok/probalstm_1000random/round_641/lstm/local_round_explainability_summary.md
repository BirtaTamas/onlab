# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `6`

## Largest probability jumps

- tick `46375`, seconds `48.50`, LSTM `0.8069`, delta `+0.2616`
- tick `46663`, seconds `53.00`, LSTM `0.8999`, delta `+0.2462`
- tick `44935`, seconds `26.00`, LSTM `0.6477`, delta `+0.1654`
- tick `46311`, seconds `47.50`, LSTM `0.5647`, delta `-0.1340`
- tick `46279`, seconds `47.00`, LSTM `0.6987`, delta `+0.1200`
- tick `48551`, seconds `82.50`, LSTM `0.8475`, delta `-0.1155`
- tick `44743`, seconds `23.00`, LSTM `0.5600`, delta `-0.0720`
- tick `46471`, seconds `50.00`, LSTM `0.7030`, delta `-0.0631`
- tick `46407`, seconds `49.00`, LSTM `0.7493`, delta `-0.0576`
- tick `45127`, seconds `29.00`, LSTM `0.5498`, delta `-0.0450`

## Top 15 local ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005173`, |coef| `0.005173`
- `lag_00__CT_kills_last_3s`: coefficient `0.004168`, |coef| `0.004168`
- `lag_00__CT_shots_fired_sum`: coefficient `0.003433`, |coef| `0.003433`
- `lag_12__CT5__is_scoped`: coefficient `-0.002317`, |coef| `0.002317`
- `lag_00__T_shots_fired_sum`: coefficient `-0.002309`, |coef| `0.002309`
- `lag_06__CT_place_TUNNELS`: coefficient `-0.002297`, |coef| `0.002297`
- `lag_00__T_kills_last_3s`: coefficient `-0.002235`, |coef| `0.002235`
- `lag_05__T2__is_walking`: coefficient `-0.002172`, |coef| `0.002172`
- `lag_14__CT1__is_walking`: coefficient `0.002091`, |coef| `0.002091`
- `lag_00__CT_damage_last_5s`: coefficient `0.002009`, |coef| `0.002009`
- `lag_12__CT5__duck_amount`: coefficient `-0.001938`, |coef| `0.001938`
- `lag_06__T2__is_walking`: coefficient `0.001883`, |coef| `0.001883`
- `lag_06__T4__is_walking`: coefficient `-0.001881`, |coef| `0.001881`
- `lag_00__damage_diff_last_5s`: coefficient `0.001855`, |coef| `0.001855`
- `lag_14__CT5__duck_amount`: coefficient `0.001699`, |coef| `0.001699`

## Top 10 utility ridge features

- `lag_05__T2__smoke`: coefficient `-0.001333` (lowers CT win probability)
- `lag_02__CT5__smoke`: coefficient `-0.001223` (lowers CT win probability)
- `lag_02__CT_B_site_active_smokes`: coefficient `-0.001118` (lowers CT win probability)
- `lag_02__CT_A_site_active_smokes`: coefficient `-0.001057` (lowers CT win probability)
- `lag_02__CT5__flash`: coefficient `-0.001003` (lowers CT win probability)
- `lag_02__CT5__utility_total`: coefficient `-0.000951` (lowers CT win probability)
- `lag_03__T3__flash`: coefficient `-0.000910` (lowers CT win probability)
- `lag_11__CT5__smoke`: coefficient `-0.000803` (lowers CT win probability)
- `lag_15__CT_A_site_active_infernos`: coefficient `-0.000799` (lowers CT win probability)
- `lag_14__T2__smoke`: coefficient `-0.000795` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__kill_diff_last_3s`: coefficient `0.005173` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.004168` (raises CT win probability)
- `lag_00__CT_shots_fired_sum`: coefficient `0.003433` (raises CT win probability)
- `lag_12__CT5__is_scoped`: coefficient `-0.002317` (lowers CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.002309` (lowers CT win probability)
- `lag_06__CT_place_TUNNELS`: coefficient `-0.002297` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.002235` (lowers CT win probability)
- `lag_05__T2__is_walking`: coefficient `-0.002172` (lowers CT win probability)
- `lag_14__CT1__is_walking`: coefficient `0.002091` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.002009` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `46375`, seconds `48.50`, LSTM delta `+0.2616`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012451`
- `lag_00__CT_kills_last_3s`: contribution `+0.012034`
- `lag_00__CT_shots_fired_sum`: contribution `+0.007155`
- `lag_06__CT_place_TUNNELS`: contribution `+0.007029`
- `lag_14__CT5__duck_amount`: contribution `+0.006413`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46663`, seconds `53.00`, LSTM delta `+0.2462`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012451`
- `lag_00__T_shots_fired_sum`: contribution `+0.012118`
- `lag_00__CT_kills_last_3s`: contribution `+0.012034`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011925`
- `lag_12__CT5__is_scoped`: contribution `+0.008286`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `44935`, seconds `26.00`, LSTM delta `+0.1654`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.024902`
- `lag_00__T_shots_fired_sum`: contribution `+0.015580`
- `lag_00__CT_kills_last_3s`: contribution `+0.012034`
- `lag_00__CT_shots_fired_sum`: contribution `+0.011925`
- `lag_00__T_kills_last_3s`: contribution `+0.007081`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46311`, seconds `47.50`, LSTM delta `-0.1340`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `-0.012451`
- `lag_12__CT5__is_scoped`: contribution `-0.008286`
- `lag_12__CT5__duck_amount`: contribution `-0.007316`
- `lag_00__T_kills_last_3s`: contribution `-0.007081`
- `lag_14__CT1__is_walking`: contribution `-0.004880`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `46279`, seconds `47.00`, LSTM delta `+0.1200`

Top all feature movements:
- `lag_00__kill_diff_last_3s`: contribution `+0.012451`
- `lag_00__CT_kills_last_3s`: contribution `+0.012034`
- `lag_12__CT5__is_scoped`: contribution `+0.008286`
- `lag_05__T2__is_walking`: contribution `+0.004988`
- `lag_11__CT5__duck_amount`: contribution `+0.004851`

Top utility-only movements:
- No utility movement among the top local contributors.
