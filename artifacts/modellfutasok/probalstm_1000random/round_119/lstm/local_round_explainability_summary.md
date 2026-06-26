# Local Round Explainability

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `15`

## Largest probability jumps

- tick `137180`, seconds `26.50`, LSTM `0.8662`, delta `+0.1517`
- tick `139548`, seconds `63.50`, LSTM `0.9557`, delta `+0.0757`
- tick `138716`, seconds `50.50`, LSTM `0.8539`, delta `-0.0706`
- tick `136636`, seconds `18.00`, LSTM `0.7082`, delta `+0.0693`
- tick `137276`, seconds `28.00`, LSTM `0.9212`, delta `+0.0293`
- tick `136924`, seconds `22.50`, LSTM `0.7433`, delta `+0.0288`
- tick `138652`, seconds `49.50`, LSTM `0.9331`, delta `+0.0279`
- tick `136668`, seconds `18.50`, LSTM `0.7332`, delta `+0.0250`
- tick `138588`, seconds `48.50`, LSTM `0.9136`, delta `+0.0248`
- tick `138812`, seconds `52.00`, LSTM `0.8618`, delta `+0.0224`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001289`, |coef| `0.001289`
- `lag_11__T1__shots_fired`: coefficient `-0.001230`, |coef| `0.001230`
- `lag_11__T_shots_fired_sum`: coefficient `-0.001203`, |coef| `0.001203`
- `lag_00__kill_diff_last_3s`: coefficient `0.001060`, |coef| `0.001060`
- `lag_07__CT_flashes_last_5s`: coefficient `0.001034`, |coef| `0.001034`
- `lag_00__T_shots_fired_sum`: coefficient `-0.000912`, |coef| `0.000912`
- `lag_00__CT_place_BANANA`: coefficient `0.000875`, |coef| `0.000875`
- `lag_00__CT_kills_last_3s`: coefficient `0.000865`, |coef| `0.000865`
- `lag_06__CT_flashes_last_5s`: coefficient `0.000669`, |coef| `0.000669`
- `lag_08__T_flash_duration_sum`: coefficient `0.000658`, |coef| `0.000658`
- `lag_08__T3__flash_duration`: coefficient `0.000641`, |coef| `0.000641`
- `lag_04__CT1__is_walking`: coefficient `0.000639`, |coef| `0.000639`
- `lag_00__damage_diff_last_5s`: coefficient `0.000631`, |coef| `0.000631`
- `lag_01__CT_shots_fired_sum`: coefficient `0.000623`, |coef| `0.000623`
- `lag_09__CT2__duck_amount`: coefficient `0.000622`, |coef| `0.000622`

## Top 10 utility ridge features

- `lag_07__CT_flashes_last_5s`: coefficient `0.001034` (raises CT win probability)
- `lag_06__CT_flashes_last_5s`: coefficient `0.000669` (raises CT win probability)
- `lag_08__T_flash_duration_sum`: coefficient `0.000658` (raises CT win probability)
- `lag_08__T3__flash_duration`: coefficient `0.000641` (raises CT win probability)
- `lag_00__T1__flash`: coefficient `-0.000515` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000495` (lowers CT win probability)
- `lag_08__T4__flash_duration`: coefficient `0.000484` (raises CT win probability)
- `lag_08__T5__flash_duration`: coefficient `0.000484` (raises CT win probability)
- `lag_00__T_smoke_inv`: coefficient `-0.000406` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000399` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.001289` (raises CT win probability)
- `lag_11__T1__shots_fired`: coefficient `-0.001230` (lowers CT win probability)
- `lag_11__T_shots_fired_sum`: coefficient `-0.001203` (lowers CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.001060` (raises CT win probability)
- `lag_00__T_shots_fired_sum`: coefficient `-0.000912` (lowers CT win probability)
- `lag_00__CT_place_BANANA`: coefficient `0.000875` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.000865` (raises CT win probability)
- `lag_04__CT1__is_walking`: coefficient `0.000639` (raises CT win probability)
- `lag_00__damage_diff_last_5s`: coefficient `0.000631` (raises CT win probability)
- `lag_01__CT_shots_fired_sum`: coefficient `0.000623` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `137180`, seconds `26.50`, LSTM delta `+0.1517`

Top all feature movements:
- `lag_11__T_shots_fired_sum`: contribution `+0.021653`
- `lag_11__T1__shots_fired`: contribution `+0.017644`
- `lag_08__T_flash_duration_sum`: contribution `+0.004508`
- `lag_00__CT_shots_fired_sum`: contribution `+0.003583`
- `lag_08__T_flashed_players`: contribution `+0.003231`

Top utility-only movements:
- `lag_08__T_flash_duration_sum`: contribution `+0.004508`
- `lag_08__T3__flash_duration`: contribution `+0.003033`
- `lag_08__T5__flash_duration`: contribution `+0.002916`
- `lag_08__T4__flash_duration`: contribution `+0.002892`
- `lag_00__T1__flash`: contribution `+0.001432`

### tick `139548`, seconds `63.50`, LSTM delta `+0.0757`

Top all feature movements:
- `lag_07__CT_flashes_last_5s`: contribution `+0.011367`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005375`
- `lag_00__kill_diff_last_3s`: contribution `+0.002551`
- `lag_00__CT_kills_last_3s`: contribution `+0.002497`
- `lag_00__T_shots_fired_sum`: contribution `+0.002051`

Top utility-only movements:
- `lag_07__CT_flashes_last_5s`: contribution `+0.011367`
- `lag_03__CT2__molly`: contribution `+0.000901`

### tick `138716`, seconds `50.50`, LSTM delta `-0.0706`

Top all feature movements:
- `lag_01__CT_shots_fired_sum`: contribution `-0.003028`
- `lag_01__T_shots_fired_sum`: contribution `-0.002931`
- `lag_00__T_shots_fired_sum`: contribution `-0.002734`
- `lag_00__CT_place_BANANA`: contribution `-0.002591`
- `lag_00__kill_diff_last_3s`: contribution `-0.002551`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `136636`, seconds `18.00`, LSTM delta `+0.0693`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.004479`
- `lag_14__CT_place_BALCONY`: contribution `+0.003006`
- `lag_12__CT_place_BALCONY`: contribution `+0.002907`
- `lag_00__kill_diff_last_3s`: contribution `+0.002551`
- `lag_00__CT_kills_last_3s`: contribution `+0.002497`

Top utility-only movements:
- `lag_01__T_active_infernos`: contribution `+0.001460`
- `lag_15__T_active_infernos`: contribution `+0.001190`
- `lag_09__CT_B_site_active_infernos`: contribution `+0.001074`
- `lag_06__CT_A_site_active_infernos`: contribution `+0.001047`

### tick `137276`, seconds `28.00`, LSTM delta `+0.0293`

Top all feature movements:
- `lag_00__CT_shots_fired_sum`: contribution `+0.006270`
- `lag_14__T_shots_fired_sum`: contribution `-0.005350`
- `lag_14__T1__shots_fired`: contribution `-0.004572`
- `lag_00__CT4__duck_amount`: contribution `+0.001828`
- `lag_15__T1__shots_fired`: contribution `+0.001660`

Top utility-only movements:
- `lag_11__T_flash_duration_sum`: contribution `+0.001305`
- `lag_01__T4__flash_duration`: contribution `+0.001283`
- `lag_01__T5__flash_duration`: contribution `+0.001208`
- `lag_11__T3__flash_duration`: contribution `+0.001126`
- `lag_01__T_flash_duration_sum`: contribution `+0.000952`
