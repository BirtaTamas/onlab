# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `11`

## Largest probability jumps

- tick `92322`, seconds `51.00`, LSTM `0.8521`, delta `+0.1339`
- tick `89794`, seconds `11.50`, LSTM `0.7594`, delta `+0.1303`
- tick `89314`, seconds `4.00`, LSTM `0.7175`, delta `+0.0469`
- tick `92546`, seconds `54.50`, LSTM `0.9598`, delta `+0.0362`
- tick `92354`, seconds `51.50`, LSTM `0.8882`, delta `+0.0361`
- tick `92194`, seconds `49.00`, LSTM `0.7159`, delta `-0.0300`
- tick `89666`, seconds `9.50`, LSTM `0.6460`, delta `-0.0275`
- tick `89218`, seconds `2.50`, LSTM `0.6595`, delta `-0.0243`
- tick `91074`, seconds `31.50`, LSTM `0.7368`, delta `+0.0238`
- tick `91714`, seconds `41.50`, LSTM `0.7206`, delta `-0.0224`

## Top 15 local ridge features

- `lag_01__CT_shots_fired_sum`: coefficient `-0.001765`, |coef| `0.001765`
- `lag_00__CT_place_MAIN`: coefficient `0.001508`, |coef| `0.001508`
- `lag_01__CT4__shots_fired`: coefficient `-0.001388`, |coef| `0.001388`
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `-0.001342`, |coef| `0.001342`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001089`, |coef| `0.001089`
- `lag_05__CT_place_FOUNTAIN`: coefficient `-0.001033`, |coef| `0.001033`
- `lag_00__CT5__is_walking`: coefficient `-0.000930`, |coef| `0.000930`
- `lag_03__CT_place_FOUNTAIN`: coefficient `-0.000879`, |coef| `0.000879`
- `lag_00__CT_kills_last_3s`: coefficient `0.000807`, |coef| `0.000807`
- `lag_05__CT_place_OUTSIDELONG`: coefficient `0.000796`, |coef| `0.000796`
- `lag_15__CT5__is_scoped`: coefficient `0.000795`, |coef| `0.000795`
- `lag_12__T_place_STREET`: coefficient `0.000788`, |coef| `0.000788`
- `lag_00__damage_diff_last_5s`: coefficient `0.000743`, |coef| `0.000743`
- `lag_04__CT_place_OUTSIDELONG`: coefficient `0.000723`, |coef| `0.000723`
- `lag_00__CT_damage_last_5s`: coefficient `0.000673`, |coef| `0.000673`

## Top 10 utility ridge features

- `lag_04__CT_B_site_active_infernos`: coefficient `-0.000558` (lowers CT win probability)
- `lag_15__CT_B_site_active_infernos`: coefficient `0.000529` (raises CT win probability)
- `lag_11__T4__flash_duration`: coefficient `-0.000498` (lowers CT win probability)
- `lag_12__T4__flash_duration`: coefficient `-0.000438` (lowers CT win probability)
- `lag_00__CT_B_site_active_infernos`: coefficient `0.000429` (raises CT win probability)
- `lag_02__T3__smoke`: coefficient `-0.000428` (lowers CT win probability)
- `lag_00__T4__molly`: coefficient `-0.000417` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000406` (lowers CT win probability)
- `lag_10__T4__flash_duration`: coefficient `-0.000403` (lowers CT win probability)
- `lag_00__T2__flash`: coefficient `-0.000375` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_01__CT_shots_fired_sum`: coefficient `-0.001765` (lowers CT win probability)
- `lag_00__CT_place_MAIN`: coefficient `0.001508` (raises CT win probability)
- `lag_01__CT4__shots_fired`: coefficient `-0.001388` (lowers CT win probability)
- `lag_15__CT_place_LOWERTUNNEL`: coefficient `-0.001342` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `-0.001089` (lowers CT win probability)
- `lag_05__CT_place_FOUNTAIN`: coefficient `-0.001033` (lowers CT win probability)
- `lag_00__CT5__is_walking`: coefficient `-0.000930` (lowers CT win probability)
- `lag_03__CT_place_FOUNTAIN`: coefficient `-0.000879` (lowers CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000807` (raises CT win probability)
- `lag_05__CT_place_OUTSIDELONG`: coefficient `0.000796` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `92322`, seconds `51.00`, LSTM delta `+0.1339`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `+0.022077`
- `lag_01__CT4__shots_fired`: contribution `+0.013456`
- `lag_00__CT_place_MAIN`: contribution `+0.010152`
- `lag_03__CT_place_FOUNTAIN`: contribution `+0.009251`
- `lag_00__CT_place_FOUNTAIN`: contribution `+0.006934`

Top utility-only movements:
- `lag_04__CT_B_site_active_infernos`: contribution `+0.001916`
- `lag_15__CT_B_site_active_infernos`: contribution `+0.001819`

### tick `89794`, seconds `11.50`, LSTM delta `+0.1303`

Top all feature movements:
- `lag_15__CT_place_LOWERTUNNEL`: contribution `+0.019733`
- `lag_00__CT_place_MAIN`: contribution `+0.010152`
- `lag_12__T_place_STREET`: contribution `+0.008664`
- `lag_05__CT_place_OUTSIDELONG`: contribution `+0.008073`
- `lag_04__CT_place_OUTSIDELONG`: contribution `+0.007338`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `89314`, seconds `4.00`, LSTM delta `+0.0469`

Top all feature movements:
- `lag_08__CT_place_CTSIDEUPPER`: contribution `+0.015258`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `+0.007497`
- `lag_05__CT_place_CTSIDEUPPER`: contribution `+0.005521`
- `lag_04__CT_place_LOWERTUNNEL`: contribution `+0.001858`
- `lag_01__CT_place_PALACEINTERIOR`: contribution `+0.001629`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `92546`, seconds `54.50`, LSTM delta `+0.0362`

Top all feature movements:
- `lag_08__CT_shots_fired_sum`: contribution `+0.003002`
- `lag_00__CT_kills_last_3s`: contribution `+0.002330`
- `lag_00__CT5__is_walking`: contribution `+0.002228`
- `lag_06__CT_place_MAIN`: contribution `+0.002180`
- `lag_08__CT4__shots_fired`: contribution `+0.001868`

Top utility-only movements:
- `lag_11__CT_B_site_active_infernos`: contribution `+0.001078`

### tick `92354`, seconds `51.50`, LSTM delta `+0.0361`

Top all feature movements:
- `lag_00__CT_place_MAIN`: contribution `+0.010152`
- `lag_04__CT_place_FOUNTAIN`: contribution `+0.006686`
- `lag_01__CT_place_FOUNTAIN`: contribution `+0.006516`
- `lag_04__CT_shots_fired_sum`: contribution `+0.002025`
- `lag_02__CT_shots_fired_sum`: contribution `-0.001921`

Top utility-only movements:
- No utility movement among the top local contributors.
