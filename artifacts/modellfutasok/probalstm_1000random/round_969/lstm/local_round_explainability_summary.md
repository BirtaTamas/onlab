# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `14`

## Largest probability jumps

- tick `119504`, seconds `21.50`, LSTM `0.2521`, delta `-0.1661`
- tick `121712`, seconds `56.00`, LSTM `0.0167`, delta `-0.0628`
- tick `118832`, seconds `11.00`, LSTM `0.3514`, delta `+0.0626`
- tick `119536`, seconds `22.00`, LSTM `0.1919`, delta `-0.0603`
- tick `119312`, seconds `18.50`, LSTM `0.4019`, delta `+0.0581`
- tick `119408`, seconds `20.00`, LSTM `0.4166`, delta `+0.0569`
- tick `120016`, seconds `29.50`, LSTM `0.1973`, delta `-0.0567`
- tick `119216`, seconds `17.00`, LSTM `0.3399`, delta `-0.0535`
- tick `118704`, seconds `9.00`, LSTM `0.3323`, delta `-0.0515`
- tick `118544`, seconds `6.50`, LSTM `0.3301`, delta `+0.0476`

## Top 15 local ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001214`, |coef| `0.001214`
- `lag_02__CT3__is_scoped`: coefficient `-0.001146`, |coef| `0.001146`
- `lag_14__T_place_DUMPSTER`: coefficient `0.001092`, |coef| `0.001092`
- `lag_00__T_place_TSTAIRS`: coefficient `-0.001036`, |coef| `0.001036`
- `lag_11__T_place_ALLEY`: coefficient `0.001003`, |coef| `0.001003`
- `lag_04__T_place_ALLEY`: coefficient `0.000999`, |coef| `0.000999`
- `lag_14__CT1__is_walking`: coefficient `0.000972`, |coef| `0.000972`
- `lag_00__CT_scoped_count`: coefficient `0.000948`, |coef| `0.000948`
- `lag_03__CT3__is_scoped`: coefficient `-0.000935`, |coef| `0.000935`
- `lag_15__CT5__flash_duration`: coefficient `-0.000925`, |coef| `0.000925`
- `lag_12__T_shots_fired_sum`: coefficient `-0.000892`, |coef| `0.000892`
- `lag_15__T_place_DUMPSTER`: coefficient `0.000869`, |coef| `0.000869`
- `lag_08__T_shots_fired_sum`: coefficient `0.000862`, |coef| `0.000862`
- `lag_11__T_place_IVY`: coefficient `-0.000861`, |coef| `0.000861`
- `lag_14__CT5__flash_duration`: coefficient `-0.000860`, |coef| `0.000860`

## Top 10 utility ridge features

- `lag_15__CT5__flash_duration`: coefficient `-0.000925` (lowers CT win probability)
- `lag_14__CT5__flash_duration`: coefficient `-0.000860` (lowers CT win probability)
- `lag_05__CT5__flash_duration`: coefficient `0.000784` (raises CT win probability)
- `lag_06__CT5__flash_duration`: coefficient `0.000771` (raises CT win probability)
- `lag_10__T_A_site_active_infernos`: coefficient `0.000708` (raises CT win probability)
- `lag_10__T_B_site_active_infernos`: coefficient `0.000674` (raises CT win probability)
- `lag_00__CT3__smoke`: coefficient `0.000646` (raises CT win probability)
- `lag_02__CT1__flash_duration`: coefficient `-0.000638` (lowers CT win probability)
- `lag_04__CT1__flash_duration`: coefficient `-0.000594` (lowers CT win probability)
- `lag_00__CT3__utility_total`: coefficient `0.000557` (raises CT win probability)

## Top 10 non-utility ridge features

- `lag_00__T_kills_last_3s`: coefficient `-0.001214` (lowers CT win probability)
- `lag_02__CT3__is_scoped`: coefficient `-0.001146` (lowers CT win probability)
- `lag_14__T_place_DUMPSTER`: coefficient `0.001092` (raises CT win probability)
- `lag_00__T_place_TSTAIRS`: coefficient `-0.001036` (lowers CT win probability)
- `lag_11__T_place_ALLEY`: coefficient `0.001003` (raises CT win probability)
- `lag_04__T_place_ALLEY`: coefficient `0.000999` (raises CT win probability)
- `lag_14__CT1__is_walking`: coefficient `0.000972` (raises CT win probability)
- `lag_00__CT_scoped_count`: coefficient `0.000948` (raises CT win probability)
- `lag_03__CT3__is_scoped`: coefficient `-0.000935` (lowers CT win probability)
- `lag_12__T_shots_fired_sum`: coefficient `-0.000892` (lowers CT win probability)

## Largest Jump Contribution Breakdown


### tick `119504`, seconds `21.50`, LSTM delta `-0.1661`

Top all feature movements:
- `lag_08__T_shots_fired_sum`: contribution `-0.007107`
- `lag_08__T3__shots_fired`: contribution `-0.005316`
- `lag_11__T_place_IVY`: contribution `-0.004599`
- `lag_04__T_place_IVY`: contribution `-0.004484`
- `lag_03__CT3__is_scoped`: contribution `-0.004253`

Top utility-only movements:
- `lag_04__CT5__flash_duration`: contribution `-0.003693`

### tick `121712`, seconds `56.00`, LSTM delta `-0.0628`

Top all feature movements:
- `lag_14__T_place_DUMPSTER`: contribution `-0.009929`
- `lag_00__T_kills_last_3s`: contribution `-0.003847`
- `lag_07__T_place_BACKOFB`: contribution `-0.002268`
- `lag_00__CT_place_TMAIN`: contribution `-0.002090`
- `lag_00__T_damage_last_5s`: contribution `-0.001916`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `118832`, seconds `11.00`, LSTM delta `+0.0626`

Top all feature movements:
- `lag_11__CT_smokes_last_5s`: contribution `+0.006054`
- `lag_13__CT_place_ENTRANCE`: contribution `+0.004348`
- `lag_06__CT3__is_scoped`: contribution `-0.003581`
- `lag_11__T_he_last_5s`: contribution `+0.003443`
- `lag_14__CT_place_ENTRANCE`: contribution `+0.003364`

Top utility-only movements:
- `lag_11__CT_smokes_last_5s`: contribution `+0.006054`
- `lag_11__T_he_last_5s`: contribution `+0.003443`
- `lag_04__CT3__flash_duration`: contribution `+0.002656`
- `lag_11__CT_flashes_last_5s`: contribution `+0.002438`
- `lag_04__CT5__flash_duration`: contribution `+0.001383`

### tick `119536`, seconds `22.00`, LSTM delta `-0.0603`

Top all feature movements:
- `lag_05__CT5__flash_duration`: contribution `-0.005201`
- `lag_06__CT3__is_scoped`: contribution `+0.003581`
- `lag_08__T_shots_fired_sum`: contribution `+0.003230`
- `lag_07__CT3__is_scoped`: contribution `-0.003070`
- `lag_10__T_shots_fired_sum`: contribution `-0.002766`

Top utility-only movements:
- `lag_05__CT5__flash_duration`: contribution `-0.005201`

### tick `119312`, seconds `18.50`, LSTM delta `+0.0581`

Top all feature movements:
- `lag_03__CT3__is_scoped`: contribution `+0.004253`
- `lag_00__CT3__is_scoped`: contribution `+0.003091`
- `lag_07__CT3__is_scoped`: contribution `+0.003070`
- `lag_05__CT2__duck_amount`: contribution `+0.002673`
- `lag_15__CT5__flash_duration`: contribution `+0.002296`

Top utility-only movements:
- `lag_15__CT5__flash_duration`: contribution `+0.002296`
