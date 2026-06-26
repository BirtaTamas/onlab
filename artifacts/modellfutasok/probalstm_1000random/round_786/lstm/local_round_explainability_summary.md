# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `21`

## Largest probability jumps

- tick `171243`, seconds `66.50`, LSTM `0.2658`, delta `-0.2395`
- tick `171275`, seconds `67.00`, LSTM `0.1110`, delta `-0.1548`
- tick `170763`, seconds `59.00`, LSTM `0.3338`, delta `+0.1004`
- tick `170795`, seconds `59.50`, LSTM `0.4254`, delta `+0.0916`
- tick `172459`, seconds `85.50`, LSTM `0.0859`, delta `-0.0539`
- tick `167531`, seconds `8.50`, LSTM `0.1592`, delta `-0.0520`
- tick `171307`, seconds `67.50`, LSTM `0.0639`, delta `-0.0471`
- tick `172363`, seconds `84.00`, LSTM `0.1005`, delta `+0.0461`
- tick `168715`, seconds `27.00`, LSTM `0.2363`, delta `+0.0437`
- tick `167403`, seconds `6.50`, LSTM `0.2424`, delta `-0.0418`

## Top 15 local ridge features

- `lag_15__T_B_site_active_infernos`: coefficient `0.002629`, |coef| `0.002629`
- `lag_14__T_B_site_active_infernos`: coefficient `0.002414`, |coef| `0.002414`
- `lag_13__T_B_site_active_infernos`: coefficient `0.002178`, |coef| `0.002178`
- `lag_12__CT_place_TRUCK`: coefficient `-0.001952`, |coef| `0.001952`
- `lag_15__T_active_infernos`: coefficient `0.001939`, |coef| `0.001939`
- `lag_14__T_active_infernos`: coefficient `0.001718`, |coef| `0.001718`
- `lag_15__T2__shots_fired`: coefficient `-0.001683`, |coef| `0.001683`
- `lag_13__T_active_infernos`: coefficient `0.001603`, |coef| `0.001603`
- `lag_13__CT_place_TRUCK`: coefficient `-0.001562`, |coef| `0.001562`
- `lag_02__T4__duck_amount`: coefficient `0.001549`, |coef| `0.001549`
- `lag_01__T4__duck_amount`: coefficient `0.001543`, |coef| `0.001543`
- `lag_12__CT_place_APARTMENTS`: coefficient `0.001513`, |coef| `0.001513`
- `lag_00__T_place_JUNGLE`: coefficient `-0.001431`, |coef| `0.001431`
- `lag_00__T_place_CONNECTOR`: coefficient `0.001397`, |coef| `0.001397`
- `lag_00__damage_diff_last_5s`: coefficient `0.001392`, |coef| `0.001392`

## Top 10 utility ridge features

- `lag_15__T_B_site_active_infernos`: coefficient `0.002629` (raises CT win probability)
- `lag_14__T_B_site_active_infernos`: coefficient `0.002414` (raises CT win probability)
- `lag_13__T_B_site_active_infernos`: coefficient `0.002178` (raises CT win probability)
- `lag_15__T_active_infernos`: coefficient `0.001939` (raises CT win probability)
- `lag_14__T_active_infernos`: coefficient `0.001718` (raises CT win probability)
- `lag_13__T_active_infernos`: coefficient `0.001603` (raises CT win probability)
- `lag_15__active_infernos_total`: coefficient `0.001340` (raises CT win probability)
- `lag_14__active_infernos_total`: coefficient `0.001190` (raises CT win probability)
- `lag_12__T_B_site_active_infernos`: coefficient `0.001146` (raises CT win probability)
- `lag_13__active_infernos_total`: coefficient `0.001113` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_12__CT_place_TRUCK`: coefficient `-0.001952` (lowers CT win probability)
- `lag_15__T2__shots_fired`: coefficient `-0.001683` (lowers CT win probability)
- `lag_13__CT_place_TRUCK`: coefficient `-0.001562` (lowers CT win probability)
- `lag_02__T4__duck_amount`: coefficient `0.001549` (raises CT win probability)
- `lag_01__T4__duck_amount`: coefficient `0.001543` (raises CT win probability)
- `lag_12__CT_place_APARTMENTS`: coefficient `0.001513` (raises CT win probability)
- `lag_00__T_place_JUNGLE`: coefficient `-0.001431` (lowers CT win probability)
- `lag_00__T_place_CONNECTOR`: coefficient `0.001397` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.001392` (raises CT win probability)
- `lag_14__CT_shots_fired_sum`: coefficient `0.001300` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `171243`, seconds `66.50`, LSTM delta `-0.2395`

Top all feature movements:
- `lag_12__CT_place_TRUCK`: contribution `-0.012589`
- `lag_15__T_B_site_active_infernos`: contribution `-0.007434`
- `lag_14__CT_shots_fired_sum`: contribution `-0.006324`
- `lag_13__T_B_site_active_infernos`: contribution `-0.006159`
- `lag_12__CT_place_APARTMENTS`: contribution `-0.005811`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `-0.007434`
- `lag_13__T_B_site_active_infernos`: contribution `-0.006159`
- `lag_15__T_active_infernos`: contribution `-0.004038`
- `lag_13__T_active_infernos`: contribution `-0.003339`

### tick `171275`, seconds `67.00`, LSTM delta `-0.1548`

Top all feature movements:
- `lag_13__CT_place_TRUCK`: contribution `-0.010076`
- `lag_14__T_B_site_active_infernos`: contribution `-0.006826`
- `lag_00__T_place_CONNECTOR`: contribution `-0.006765`
- `lag_02__T4__duck_amount`: contribution `-0.005727`
- `lag_13__CT_place_APARTMENTS`: contribution `-0.004512`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `-0.006826`
- `lag_14__T_active_infernos`: contribution `-0.003579`

### tick `170763`, seconds `59.00`, LSTM delta `+0.1004`

Top all feature movements:
- `lag_14__T_B_site_active_infernos`: contribution `+0.006826`
- `lag_01__T4__duck_amount`: contribution `+0.005705`
- `lag_02__T2__duck_amount`: contribution `+0.004263`
- `lag_05__T4__is_scoped`: contribution `+0.003828`
- `lag_14__T_active_infernos`: contribution `+0.003579`

Top utility-only movements:
- `lag_14__T_B_site_active_infernos`: contribution `+0.006826`
- `lag_14__T_active_infernos`: contribution `+0.003579`
- `lag_12__T_B_site_active_infernos`: contribution `+0.003241`
- `lag_12__T_active_infernos`: contribution `+0.001912`
- `lag_14__active_infernos_total`: contribution `+0.001709`

### tick `170795`, seconds `59.50`, LSTM delta `+0.0916`

Top all feature movements:
- `lag_15__T_B_site_active_infernos`: contribution `+0.007434`
- `lag_13__T_B_site_active_infernos`: contribution `+0.006159`
- `lag_02__T4__duck_amount`: contribution `+0.005727`
- `lag_15__T_active_infernos`: contribution `+0.004038`
- `lag_13__T_active_infernos`: contribution `+0.003339`

Top utility-only movements:
- `lag_15__T_B_site_active_infernos`: contribution `+0.007434`
- `lag_13__T_B_site_active_infernos`: contribution `+0.006159`
- `lag_15__T_active_infernos`: contribution `+0.004038`
- `lag_13__T_active_infernos`: contribution `+0.003339`
- `lag_15__active_infernos_total`: contribution `+0.001925`

### tick `172459`, seconds `85.50`, LSTM delta `-0.0539`

Top all feature movements:
- `lag_03__CT_place_PALACEALLEY`: contribution `-0.011472`
- `lag_07__T_place_STAIRS`: contribution `-0.009686`
- `lag_03__CT_place_TSPAWN`: contribution `-0.003196`
- `lag_03__T1__duck_amount`: contribution `+0.002436`
- `lag_15__T_place_CTSPAWN`: contribution `-0.002054`

Top utility-only movements:
- No utility movement among the top local contributors.
