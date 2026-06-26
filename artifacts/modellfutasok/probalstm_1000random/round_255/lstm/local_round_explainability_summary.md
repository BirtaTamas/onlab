# Local Round Explainability

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-mouz-vs-falcons-bo3-ET1FlQ7LAGQtcSrRzzPcv6/mouz-vs-falcons-m1-dust2.csv`
- round_num: `12`

## Largest probability jumps

- tick `91689`, seconds `38.00`, LSTM `0.8713`, delta `+0.1613`
- tick `93065`, seconds `59.50`, LSTM `0.9117`, delta `+0.1220`
- tick `92969`, seconds `58.00`, LSTM `0.8245`, delta `-0.0750`
- tick `92425`, seconds `49.50`, LSTM `0.9183`, delta `+0.0464`
- tick `92393`, seconds `49.00`, LSTM `0.8719`, delta `+0.0333`
- tick `89993`, seconds `11.50`, LSTM `0.7086`, delta `+0.0314`
- tick `93001`, seconds `58.50`, LSTM `0.7943`, delta `-0.0303`
- tick `90569`, seconds `20.50`, LSTM `0.6557`, delta `-0.0290`
- tick `91401`, seconds `33.50`, LSTM `0.7043`, delta `+0.0279`
- tick `93129`, seconds `60.50`, LSTM `0.9605`, delta `+0.0261`

## Top 15 local ridge features

- `lag_09__T1__flash_duration`: coefficient `0.001423`, |coef| `0.001423`
- `lag_00__CT_kills_last_3s`: coefficient `0.001392`, |coef| `0.001392`
- `lag_00__kill_diff_last_3s`: coefficient `0.001383`, |coef| `0.001383`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001375`, |coef| `0.001375`
- `lag_06__CT5__flash_duration`: coefficient `0.001310`, |coef| `0.001310`
- `lag_00__CT4__is_scoped`: coefficient `-0.001215`, |coef| `0.001215`
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `-0.001034`, |coef| `0.001034`
- `lag_15__CT_shots_fired_sum`: coefficient `-0.000997`, |coef| `0.000997`
- `lag_03__CT2__is_scoped`: coefficient `0.000929`, |coef| `0.000929`
- `lag_06__T_place_TUNNELSTAIRS`: coefficient `-0.000875`, |coef| `0.000875`
- `lag_14__T5__duck_amount`: coefficient `0.000846`, |coef| `0.000846`
- `lag_00__damage_diff_last_5s`: coefficient `0.000844`, |coef| `0.000844`
- `lag_00__CT_place_HOLE`: coefficient `0.000810`, |coef| `0.000810`
- `lag_09__CT_place_HOLE`: coefficient `0.000772`, |coef| `0.000772`
- `lag_00__CT4__is_walking`: coefficient `-0.000768`, |coef| `0.000768`

## Top 10 utility ridge features

- `lag_09__T1__flash_duration`: coefficient `0.001423` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.001310` (raises CT win probability)
- `lag_15__CT4__flash_duration`: coefficient `-0.000759` (lowers CT win probability)
- `lag_09__T_flash_duration_sum`: coefficient `0.000719` (raises CT win probability)
- `lag_15__CT_flash_duration_sum`: coefficient `-0.000699` (lowers CT win probability)
- `lag_15__CT_active_infernos`: coefficient `0.000675` (raises CT win probability)
- `lag_06__CT4__flash_duration`: coefficient `-0.000657` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000654` (lowers CT win probability)
- `lag_15__active_infernos_total`: coefficient `0.000637` (raises CT win probability)
- `lag_03__CT_flash_duration_sum`: coefficient `0.000466` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_kills_last_3s`: coefficient `0.001392` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001383` (raises CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001375` (raises CT win probability)
- `lag_00__CT4__is_scoped`: coefficient `-0.001215` (lowers CT win probability)
- `lag_05__CT_place_SHORTSTAIRS`: coefficient `-0.001034` (lowers CT win probability)
- `lag_15__CT_shots_fired_sum`: coefficient `-0.000997` (lowers CT win probability)
- `lag_03__CT2__is_scoped`: coefficient `0.000929` (raises CT win probability)
- `lag_06__T_place_TUNNELSTAIRS`: coefficient `-0.000875` (lowers CT win probability)
- `lag_14__T5__duck_amount`: coefficient `0.000846` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000844` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `91689`, seconds `38.00`, LSTM delta `+0.1613`

Top all feature movements:
- `lag_09__T1__flash_duration`: contribution `+0.010207`
- `lag_06__CT5__flash_duration`: contribution `+0.009239`
- `lag_06__T_place_TUNNELSTAIRS`: contribution `+0.006112`
- `lag_05__CT_place_SHORTSTAIRS`: contribution `+0.005766`
- `lag_13__CT_place_SHORTSTAIRS`: contribution `+0.004243`

Top utility-only movements:
- `lag_09__T1__flash_duration`: contribution `+0.010207`
- `lag_06__CT5__flash_duration`: contribution `+0.009239`
- `lag_09__T_flash_duration_sum`: contribution `+0.002661`

### tick `93065`, seconds `59.50`, LSTM delta `+0.1220`

Top all feature movements:
- `lag_00__CT_place_OUTSIDELONG`: contribution `+0.013951`
- `lag_15__CT_shots_fired_sum`: contribution `+0.008311`
- `lag_03__CT2__is_scoped`: contribution `+0.005688`
- `lag_15__CT2__shots_fired`: contribution `+0.004280`
- `lag_00__CT4__is_scoped`: contribution `+0.004142`

Top utility-only movements:
- `lag_06__CT4__flash_duration`: contribution `+0.003106`
- `lag_06__CT2__flash_duration`: contribution `+0.002259`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001579`

### tick `92969`, seconds `58.00`, LSTM delta `-0.0750`

Top all feature movements:
- `lag_15__CT4__flash_duration`: contribution `-0.004435`
- `lag_12__CT2__is_scoped`: contribution `-0.003735`
- `lag_12__CT_shots_fired_sum`: contribution `-0.003688`
- `lag_15__CT_flash_duration_sum`: contribution `-0.003541`
- `lag_00__kill_diff_last_3s`: contribution `-0.003328`

Top utility-only movements:
- `lag_15__CT4__flash_duration`: contribution `-0.004435`
- `lag_15__CT_flash_duration_sum`: contribution `-0.003541`
- `lag_15__CT2__flash_duration`: contribution `-0.002491`
- `lag_03__CT_flash_duration_sum`: contribution `-0.002026`
- `lag_03__CT2__flash_duration`: contribution `-0.002009`

### tick `92425`, seconds `49.50`, LSTM delta `+0.0464`

Top all feature movements:
- `lag_00__CT4__is_scoped`: contribution `+0.004142`
- `lag_00__CT_kills_last_3s`: contribution `+0.004019`
- `lag_00__kill_diff_last_3s`: contribution `+0.003328`
- `lag_14__T5__duck_amount`: contribution `+0.003213`
- `lag_11__CT_place_EXTENDEDA`: contribution `+0.002908`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `92393`, seconds `49.00`, LSTM delta `+0.0333`

Top all feature movements:
- `lag_03__CT2__is_scoped`: contribution `+0.005688`
- `lag_11__CT_place_EXTENDEDA`: contribution `-0.002908`
- `lag_08__CT2__is_scoped`: contribution `+0.002778`
- `lag_00__CT_shots_fired_sum`: contribution `+0.002550`
- `lag_10__T_place_LOWERTUNNEL`: contribution `+0.002302`

Top utility-only movements:
- `lag_15__CT_flash_duration_sum`: contribution `+0.002273`
- `lag_15__CT5__flash_duration`: contribution `+0.001764`
