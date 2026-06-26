# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `15`

## Largest probability jumps

- tick `127591`, seconds `73.00`, LSTM `0.1080`, delta `-0.2135`
- tick `127143`, seconds `66.00`, LSTM `0.4010`, delta `-0.1091`
- tick `127175`, seconds `66.50`, LSTM `0.3531`, delta `-0.0480`
- tick `127335`, seconds `69.00`, LSTM `0.3670`, delta `+0.0424`
- tick `127623`, seconds `73.50`, LSTM `0.0664`, delta `-0.0416`
- tick `127559`, seconds `72.50`, LSTM `0.3215`, delta `-0.0385`
- tick `124519`, seconds `25.00`, LSTM `0.5559`, delta `+0.0320`
- tick `127271`, seconds `68.00`, LSTM `0.3147`, delta `-0.0310`
- tick `125575`, seconds `41.50`, LSTM `0.5623`, delta `+0.0256`
- tick `126759`, seconds `60.00`, LSTM `0.5488`, delta `+0.0256`

## Top 15 local ridge features

- `lag_02__CT_place_MAIN`: coefficient `-0.001927`, |coef| `0.001927`
- `lag_02__CT_place_OUTSIDELONG`: coefficient `-0.001816`, |coef| `0.001816`
- `lag_03__CT_place_OUTSIDELONG`: coefficient `-0.001645`, |coef| `0.001645`
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `-0.001571`, |coef| `0.001571`
- `lag_00__T_kills_last_3s`: coefficient `-0.001553`, |coef| `0.001553`
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001447`, |coef| `0.001447`
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001438`, |coef| `0.001438`
- `lag_03__CT_place_MAIN`: coefficient `-0.001327`, |coef| `0.001327`
- `lag_12__CT_place_TUNNEL`: coefficient `0.001246`, |coef| `0.001246`
- `lag_00__CT_place_MAIN`: coefficient `-0.001215`, |coef| `0.001215`
- `lag_09__CT_place_LOWERTUNNEL`: coefficient `0.001215`, |coef| `0.001215`
- `lag_00__T_damage_last_5s`: coefficient `-0.001208`, |coef| `0.001208`
- `lag_01__T_place_CONNECTOR`: coefficient `-0.001143`, |coef| `0.001143`
- `lag_01__CT_place_MAIN`: coefficient `-0.001134`, |coef| `0.001134`
- `lag_00__damage_diff_last_5s`: coefficient `0.001123`, |coef| `0.001123`

## Top 10 utility ridge features

- `lag_01__CT4__flash_duration`: coefficient `-0.001094` (lowers CT win probability)
- `lag_00__CT4__molly`: coefficient `0.001027` (raises CT win probability)
- `lag_14__CT1__molly`: coefficient `0.000940` (raises CT win probability)
- `lag_00__CT4__utility_total`: coefficient `0.000856` (raises CT win probability)
- `lag_00__CT4__flash_duration`: coefficient `0.000845` (raises CT win probability)
- `lag_14__CT1__utility_total`: coefficient `0.000763` (raises CT win probability)
- `lag_00__CT_molly_inv`: coefficient `0.000760` (raises CT win probability)
- `lag_00__CT1__molly`: coefficient `0.000754` (raises CT win probability)
- `lag_04__CT3__flash`: coefficient `0.000723` (raises CT win probability)
- `lag_00__CT4__flash`: coefficient `0.000723` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_02__CT_place_MAIN`: coefficient `-0.001927` (lowers CT win probability)
- `lag_02__CT_place_OUTSIDELONG`: coefficient `-0.001816` (lowers CT win probability)
- `lag_03__CT_place_OUTSIDELONG`: coefficient `-0.001645` (lowers CT win probability)
- `lag_13__CT_place_LOWERTUNNEL`: coefficient `-0.001571` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001553` (lowers CT win probability)
- `lag_00__CT_place_CTSIDEUPPER`: coefficient `0.001447` (raises CT win probability)
- `lag_00__CT_place_OUTSIDELONG`: coefficient `0.001438` (raises CT win probability)
- `lag_03__CT_place_MAIN`: coefficient `-0.001327` (lowers CT win probability)
- `lag_12__CT_place_TUNNEL`: coefficient `0.001246` (raises CT win probability)
- `lag_00__CT_place_MAIN`: coefficient `-0.001215` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `127591`, seconds `73.00`, LSTM delta `-0.2135`

Top all feature movements:
- `lag_02__CT_place_OUTSIDELONG`: contribution `-0.018420`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.014581`
- `lag_02__CT_place_MAIN`: contribution `-0.012978`
- `lag_13__CT_place_LOWERTUNNEL`: contribution `-0.011548`
- `lag_10__CT_place_OUTSIDELONG`: contribution `-0.010170`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `-0.006275`
- `lag_00__CT4__flash_duration`: contribution `-0.004846`
- `lag_00__CT4__molly`: contribution `-0.002529`
- `lag_14__CT1__molly`: contribution `-0.002340`

### tick `127143`, seconds `66.00`, LSTM delta `-0.1091`

Top all feature movements:
- `lag_12__CT_place_TUNNEL`: contribution `-0.020017`
- `lag_03__CT_place_OUTSIDELONG`: contribution `-0.016682`
- `lag_00__CT_place_OUTSIDELONG`: contribution `-0.014581`
- `lag_00__T_kills_last_3s`: contribution `-0.004918`
- `lag_00__T_damage_last_5s`: contribution `-0.002897`

Top utility-only movements:
- `lag_00__CT1__molly`: contribution `-0.001877`
- `lag_04__CT5__smoke`: contribution `-0.001324`
- `lag_00__CT1__utility_total`: contribution `-0.001254`

### tick `127175`, seconds `66.50`, LSTM delta `-0.0480`

Top all feature movements:
- `lag_04__CT_place_OUTSIDELONG`: contribution `-0.009167`
- `lag_13__CT_place_TUNNEL`: contribution `-0.007570`
- `lag_00__CT_place_CTSIDEUPPER`: contribution `-0.007471`
- `lag_00__CT_place_LOWERTUNNEL`: contribution `-0.007148`
- `lag_15__T5__duck_amount`: contribution `-0.002340`

Top utility-only movements:
- `lag_01__CT1__molly`: contribution `-0.001104`
- `lag_05__CT5__smoke`: contribution `-0.001006`

### tick `127335`, seconds `69.00`, LSTM delta `+0.0424`

Top all feature movements:
- `lag_02__CT_place_OUTSIDELONG`: contribution `+0.018420`
- `lag_04__CT_place_OUTSIDELONG`: contribution `-0.009167`
- `lag_01__CT_place_LOWERTUNNEL`: contribution `+0.005186`
- `lag_09__CT_place_OUTSIDELONG`: contribution `+0.004942`
- `lag_00__T_kills_last_3s`: contribution `+0.004918`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `127623`, seconds `73.50`, LSTM delta `-0.0416`

Top all feature movements:
- `lag_03__CT_place_OUTSIDELONG`: contribution `-0.016682`
- `lag_03__CT_place_MAIN`: contribution `-0.008938`
- `lag_01__CT4__flash_duration`: contribution `+0.006275`
- `lag_00__T_place_CONNECTOR`: contribution `-0.004742`
- `lag_15__CT_place_OUTSIDELONG`: contribution `+0.004538`

Top utility-only movements:
- `lag_01__CT4__flash_duration`: contribution `+0.006275`
- `lag_01__CT_flash_duration_sum`: contribution `+0.001349`
- `lag_15__CT1__molly`: contribution `-0.001143`
