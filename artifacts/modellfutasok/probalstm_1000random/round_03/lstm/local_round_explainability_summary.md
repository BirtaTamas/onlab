# Local Round Explainability

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `12`

## Largest probability jumps

- tick `109769`, seconds `69.50`, LSTM `0.7665`, delta `+0.1473`
- tick `110185`, seconds `76.00`, LSTM `0.8496`, delta `+0.1190`
- tick `111145`, seconds `91.00`, LSTM `0.9421`, delta `+0.0497`
- tick `106089`, seconds `12.00`, LSTM `0.6761`, delta `+0.0342`
- tick `110217`, seconds `76.50`, LSTM `0.8809`, delta `+0.0314`
- tick `106665`, seconds `21.00`, LSTM `0.6074`, delta `+0.0309`
- tick `109161`, seconds `60.00`, LSTM `0.6088`, delta `-0.0292`
- tick `105993`, seconds `10.50`, LSTM `0.6395`, delta `+0.0253`
- tick `111209`, seconds `92.00`, LSTM `0.9723`, delta `+0.0225`
- tick `109833`, seconds `70.50`, LSTM `0.7628`, delta `-0.0204`

## Top 15 local ridge features

- `lag_11__CT_place_VENTS`: coefficient `-0.002852`, |coef| `0.002852`
- `lag_00__CT_kills_last_3s`: coefficient `0.001904`, |coef| `0.001904`
- `lag_00__CT_damage_last_5s`: coefficient `0.001597`, |coef| `0.001597`
- `lag_00__damage_diff_last_5s`: coefficient `0.001578`, |coef| `0.001578`
- `lag_00__kill_diff_last_3s`: coefficient `0.001507`, |coef| `0.001507`
- `lag_11__CT2__duck_amount`: coefficient `-0.001490`, |coef| `0.001490`
- `lag_11__CT_place_TUNNELS`: coefficient `0.001359`, |coef| `0.001359`
- `lag_13__T4__duck_amount`: coefficient `-0.001283`, |coef| `0.001283`
- `lag_00__CT_place_VENTS`: coefficient `-0.001247`, |coef| `0.001247`
- `lag_00__T4__alive`: coefficient `-0.001100`, |coef| `0.001100`
- `lag_00__T4__hp`: coefficient `-0.001079`, |coef| `0.001079`
- `lag_02__CT1__duck_amount`: coefficient `-0.001063`, |coef| `0.001063`
- `lag_00__T4__armor`: coefficient `-0.001026`, |coef| `0.001026`
- `lag_12__CT_place_VENTS`: coefficient `-0.001011`, |coef| `0.001011`
- `lag_10__CT_place_VENTS`: coefficient `-0.001000`, |coef| `0.001000`

## Top 10 utility ridge features

- `lag_06__T4__molly`: coefficient `-0.000948` (lowers CT win probability)
- `lag_02__T_active_infernos`: coefficient `0.000800` (raises CT win probability)
- `lag_12__T_A_site_active_smokes`: coefficient `0.000654` (raises CT win probability)
- `lag_02__active_infernos_total`: coefficient `0.000554` (raises CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.000481` (raises CT win probability)
- `lag_12__T_active_smokes`: coefficient `0.000470` (raises CT win probability)
- `lag_05__CT1__flash_duration`: coefficient `0.000405` (raises CT win probability)
- `lag_05__T4__molly`: coefficient `-0.000397` (lowers CT win probability)
- `lag_07__T4__molly`: coefficient `-0.000383` (lowers CT win probability)
- `lag_06__T4__utility_total`: coefficient `-0.000363` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_11__CT_place_VENTS`: coefficient `-0.002852` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001904` (raises CT win probability)
- `lag_00__CT_damage_last_5s`: coefficient `0.001597` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001578` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001507` (raises CT win probability)
- `lag_11__CT2__duck_amount`: coefficient `-0.001490` (lowers CT win probability)
- `lag_11__CT_place_TUNNELS`: coefficient `0.001359` (raises CT win probability)
- `lag_13__T4__duck_amount`: coefficient `-0.001283` (lowers CT win probability)
- `lag_00__CT_place_VENTS`: coefficient `-0.001247` (lowers CT win probability)
- `lag_00__T4__alive`: coefficient `-0.001100` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `109769`, seconds `69.50`, LSTM delta `+0.1473`

Top all feature movements:
- `lag_11__CT_place_VENTS`: contribution `+0.023932`
- `lag_11__CT2__duck_amount`: contribution `+0.005678`
- `lag_00__CT_kills_last_3s`: contribution `+0.005497`
- `lag_13__T4__duck_amount`: contribution `+0.004745`
- `lag_11__CT_place_TUNNELS`: contribution `+0.004160`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `110185`, seconds `76.00`, LSTM delta `+0.1190`

Top all feature movements:
- `lag_11__CT2__duck_amount`: contribution `+0.005678`
- `lag_00__CT_kills_last_3s`: contribution `+0.005497`
- `lag_02__T_place_VENDING`: contribution `+0.004325`
- `lag_10__T_place_VENDING`: contribution `+0.003957`
- `lag_00__kill_diff_last_3s`: contribution `+0.003628`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `111145`, seconds `91.00`, LSTM delta `+0.0497`

Top all feature movements:
- `lag_11__CT2__duck_amount`: contribution `+0.005678`
- `lag_00__CT_kills_last_3s`: contribution `+0.005497`
- `lag_00__kill_diff_last_3s`: contribution `+0.003628`
- `lag_00__damage_diff_last_5s`: contribution `+0.003560`
- `lag_00__CT_damage_last_5s`: contribution `+0.003482`

Top utility-only movements:
- `lag_06__T_A_site_active_infernos`: contribution `+0.000978`
- `lag_06__T_B_site_active_infernos`: contribution `+0.000885`
- `lag_06__T_active_infernos`: contribution `+0.000698`

### tick `106089`, seconds `12.00`, LSTM delta `+0.0342`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.003324`
- `lag_01__CT1__duck_amount`: contribution `-0.003083`
- `lag_09__CT_place_HELL`: contribution `-0.003033`
- `lag_01__CT_shots_fired_sum`: contribution `+0.002203`
- `lag_00__CT_place_RAFTERS`: contribution `+0.001909`

Top utility-only movements:
- `lag_02__T_active_infernos`: contribution `+0.001666`

### tick `110217`, seconds `76.50`, LSTM delta `+0.0314`

Top all feature movements:
- `lag_00__T_place_TROPHY`: contribution `+0.006503`
- `lag_04__T_place_CONTROL`: contribution `+0.004203`
- `lag_00__CT_shots_fired_sum`: contribution `-0.003324`
- `lag_11__T_place_VENDING`: contribution `+0.003278`
- `lag_12__CT2__duck_amount`: contribution `-0.003111`

Top utility-only movements:
- No utility movement among the top local contributors.
