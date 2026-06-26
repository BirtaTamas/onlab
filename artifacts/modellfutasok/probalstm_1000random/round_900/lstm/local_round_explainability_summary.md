# Local Round Explainability

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `16`

## Largest probability jumps

- tick `140556`, seconds `38.00`, LSTM `0.5120`, delta `+0.2497`
- tick `141324`, seconds `50.00`, LSTM `0.9218`, delta `+0.1942`
- tick `140748`, seconds `41.00`, LSTM `0.6904`, delta `+0.1849`
- tick `140972`, seconds `44.50`, LSTM `0.9246`, delta `+0.1438`
- tick `141260`, seconds `49.00`, LSTM `0.7117`, delta `-0.1401`
- tick `139436`, seconds `20.50`, LSTM `0.3795`, delta `-0.0785`
- tick `140780`, seconds `41.50`, LSTM `0.7490`, delta `+0.0585`
- tick `140396`, seconds `35.50`, LSTM `0.2737`, delta `-0.0582`
- tick `140364`, seconds `35.00`, LSTM `0.3320`, delta `-0.0516`
- tick `141036`, seconds `45.50`, LSTM `0.8810`, delta `-0.0447`

## Top 15 local ridge features

- `lag_00__T_place_SIDEHALL`: coefficient `-0.003601`, |coef| `0.003601`
- `lag_12__T_place_SIDEHALL`: coefficient `0.002542`, |coef| `0.002542`
- `lag_08__T_place_SIDEHALL`: coefficient `0.001701`, |coef| `0.001701`
- `lag_09__CT_place_HOUSE`: coefficient `-0.001692`, |coef| `0.001692`
- `lag_00__kill_diff_last_3s`: coefficient `0.001593`, |coef| `0.001593`
- `lag_00__CT_kills_last_3s`: coefficient `0.001590`, |coef| `0.001590`
- `lag_00__damage_diff_last_5s`: coefficient `0.001577`, |coef| `0.001577`
- `lag_06__T_place_SIDEHALL`: coefficient `0.001530`, |coef| `0.001530`
- `lag_02__CT_place_TSIDEUPPER`: coefficient `0.001506`, |coef| `0.001506`
- `lag_14__T_place_SIDEHALL`: coefficient `0.001423`, |coef| `0.001423`
- `lag_05__T_place_SIDEHALL`: coefficient `0.001398`, |coef| `0.001398`
- `lag_01__T_place_SIDEHALL`: coefficient `-0.001392`, |coef| `0.001392`
- `lag_00__CT_damage_last_5s`: coefficient `0.001375`, |coef| `0.001375`
- `lag_08__CT_place_TSIDEUPPER`: coefficient `0.001370`, |coef| `0.001370`
- `lag_04__CT1__is_walking`: coefficient `-0.001322`, |coef| `0.001322`

## Top 10 utility ridge features

- `lag_00__CT_A_site_active_infernos`: coefficient `0.000708` (raises CT win probability)
- `lag_06__CT_A_site_active_infernos`: coefficient `0.000641` (raises CT win probability)
- `lag_11__T_A_site_active_infernos`: coefficient `-0.000623` (lowers CT win probability)
- `lag_01__T_A_site_active_infernos`: coefficient `0.000618` (raises CT win probability)
- `lag_07__T_A_site_active_infernos`: coefficient `0.000614` (raises CT win probability)
- `lag_02__CT1__molly`: coefficient `-0.000612` (lowers CT win probability)
- `lag_11__CT_A_site_active_infernos`: coefficient `0.000598` (raises CT win probability)
- `lag_05__T2__molly`: coefficient `-0.000558` (lowers CT win probability)
- `lag_00__T5__smoke`: coefficient `-0.000552` (lowers CT win probability)
- `lag_13__CT5__flash_duration`: coefficient `0.000537` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_place_SIDEHALL`: coefficient `-0.003601` (lowers CT win probability)
- `lag_12__T_place_SIDEHALL`: coefficient `0.002542` (raises CT win probability)
- `lag_08__T_place_SIDEHALL`: coefficient `0.001701` (raises CT win probability)
- `lag_09__CT_place_HOUSE`: coefficient `-0.001692` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001593` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001590` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001577` (raises CT win probability)
- `lag_06__T_place_SIDEHALL`: coefficient `0.001530` (raises CT win probability)
- `lag_02__CT_place_TSIDEUPPER`: coefficient `0.001506` (raises CT win probability)
- `lag_14__T_place_SIDEHALL`: coefficient `0.001423` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `140556`, seconds `38.00`, LSTM delta `+0.2497`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.023338`
- `lag_02__CT_place_TSIDEUPPER`: contribution `+0.011323`
- `lag_08__T_place_SIDEHALL`: contribution `+0.011025`
- `lag_06__T_place_SIDEHALL`: contribution `+0.009916`
- `lag_05__T_place_SIDEHALL`: contribution `+0.009063`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141324`, seconds `50.00`, LSTM delta `+0.1942`

Top all feature movements:
- `lag_09__CT_place_HOUSE`: contribution `+0.011955`
- `lag_05__T_place_SIDEHALL`: contribution `-0.009063`
- `lag_10__CT2__duck_amount`: contribution `+0.004902`
- `lag_15__CT_place_TSIDEUPPER`: contribution `+0.004727`
- `lag_00__CT_kills_last_3s`: contribution `+0.004591`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `140748`, seconds `41.00`, LSTM delta `+0.1849`

Top all feature movements:
- `lag_12__T_place_SIDEHALL`: contribution `+0.016474`
- `lag_08__CT_place_TSIDEUPPER`: contribution `+0.010300`
- `lag_06__T_place_SIDEHALL`: contribution `-0.009916`
- `lag_14__T_place_SIDEHALL`: contribution `+0.009220`
- `lag_05__T_place_SIDEHALL`: contribution `+0.009063`

Top utility-only movements:
- `lag_06__CT_A_site_active_infernos`: contribution `+0.002261`

### tick `140972`, seconds `44.50`, LSTM delta `+0.1438`

Top all feature movements:
- `lag_00__T_place_SIDEHALL`: contribution `+0.023338`
- `lag_12__T_place_SIDEHALL`: contribution `+0.016474`
- `lag_13__T_place_SIDEHALL`: contribution `-0.005654`
- `lag_15__CT_place_TSIDEUPPER`: contribution `-0.004727`
- `lag_04__CT_place_TSIDEUPPER`: contribution `+0.004623`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `141260`, seconds `49.00`, LSTM delta `-0.1401`

Top all feature movements:
- `lag_13__CT_place_TSIDEUPPER`: contribution `-0.009579`
- `lag_09__T_place_SIDEHALL`: contribution `-0.006807`
- `lag_09__CT_place_HOUSE`: contribution `-0.005978`
- `lag_07__CT_place_HOUSE`: contribution `-0.005048`
- `lag_00__kill_diff_last_3s`: contribution `-0.003835`

Top utility-only movements:
- `lag_11__CT_A_site_active_infernos`: contribution `-0.002111`
- `lag_09__T_A_site_active_infernos`: contribution `-0.001598`
