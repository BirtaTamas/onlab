# Local Round Explainability

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `1`

## Largest probability jumps

- tick `5242`, seconds `41.00`, LSTM `0.2692`, delta `-0.2331`
- tick `5434`, seconds `44.00`, LSTM `0.0659`, delta `-0.1346`
- tick `5786`, seconds `49.50`, LSTM `0.0234`, delta `-0.0649`
- tick `5274`, seconds `41.50`, LSTM `0.2191`, delta `-0.0501`
- tick `5658`, seconds `47.50`, LSTM `0.0653`, delta `+0.0387`
- tick `5210`, seconds `40.50`, LSTM `0.5023`, delta `-0.0207`
- tick `5754`, seconds `49.00`, LSTM `0.0883`, delta `+0.0202`
- tick `5946`, seconds `52.00`, LSTM `0.0065`, delta `-0.0126`
- tick `5466`, seconds `44.50`, LSTM `0.0548`, delta `-0.0111`
- tick `5338`, seconds `42.50`, LSTM `0.2018`, delta `-0.0106`

## Top 15 local ridge features

- `lag_13__CT_place_OUTSIDELONG`: coefficient `-0.003004`, |coef| `0.003004`
- `lag_11__CT1__duck_amount`: coefficient `-0.001763`, |coef| `0.001763`
- `lag_00__T_damage_last_5s`: coefficient `-0.001696`, |coef| `0.001696`
- `lag_00__T_kills_last_3s`: coefficient `-0.001674`, |coef| `0.001674`
- `lag_01__T_A_site_active_infernos`: coefficient `-0.001625`, |coef| `0.001625`
- `lag_10__CT3__duck_amount`: coefficient `-0.001621`, |coef| `0.001621`
- `lag_01__T_place_TSIDEUPPER`: coefficient `0.001590`, |coef| `0.001590`
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001577`, |coef| `0.001577`
- `lag_00__CT2__alive`: coefficient `0.001519`, |coef| `0.001519`
- `lag_03__CT4__is_walking`: coefficient `0.001510`, |coef| `0.001510`
- `lag_00__CT2__hp`: coefficient `0.001501`, |coef| `0.001501`
- `lag_02__CT3__duck_amount`: coefficient `0.001469`, |coef| `0.001469`
- `lag_10__T_place_CANAL`: coefficient `0.001427`, |coef| `0.001427`
- `lag_00__CT2__armor`: coefficient `0.001422`, |coef| `0.001422`
- `lag_08__CT3__duck_amount`: coefficient `0.001384`, |coef| `0.001384`

## Top 10 utility ridge features

- `lag_01__T_A_site_active_infernos`: coefficient `-0.001625` (lowers CT win probability)
- `lag_03__T_A_site_active_infernos`: coefficient `-0.001577` (lowers CT win probability)
- `lag_07__T3__molly`: coefficient `0.001314` (raises CT win probability)
- `lag_04__T4__molly`: coefficient `0.001301` (raises CT win probability)
- `lag_07__T4__smoke`: coefficient `0.001289` (raises CT win probability)
- `lag_01__T_active_infernos`: coefficient `-0.001125` (lowers CT win probability)
- `lag_03__T_active_infernos`: coefficient `-0.001096` (lowers CT win probability)
- `lag_00__T_flash_duration_sum`: coefficient `-0.001040` (lowers CT win probability)
- `lag_02__CT4__flash`: coefficient `0.001030` (raises CT win probability)
- `lag_00__T2__flash_duration`: coefficient `-0.000992` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_13__CT_place_OUTSIDELONG`: coefficient `-0.003004` (lowers CT win probability)
- `lag_11__CT1__duck_amount`: coefficient `-0.001763` (lowers CT win probability)
- `lag_00__T_damage_last_5s`: coefficient `-0.001696` (lowers CT win probability)
- `lag_00__T_kills_last_3s`: coefficient `-0.001674` (lowers CT win probability)
- `lag_10__CT3__duck_amount`: coefficient `-0.001621` (lowers CT win probability)
- `lag_01__T_place_TSIDEUPPER`: coefficient `0.001590` (raises CT win probability)
- `lag_00__CT2__alive`: coefficient `0.001519` (raises CT win probability)
- `lag_03__CT4__is_walking`: coefficient `0.001510` (raises CT win probability)
- `lag_00__CT2__hp`: coefficient `0.001501` (raises CT win probability)
- `lag_02__CT3__duck_amount`: coefficient `0.001469` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `5242`, seconds `41.00`, LSTM delta `-0.2331`

Top all feature movements:
- `lag_13__CT_place_OUTSIDELONG`: contribution `-0.030464`
- `lag_11__CT1__duck_amount`: contribution `-0.006728`
- `lag_10__CT3__duck_amount`: contribution `-0.006031`
- `lag_02__CT3__duck_amount`: contribution `-0.005467`
- `lag_00__T_kills_last_3s`: contribution `-0.005303`

Top utility-only movements:
- `lag_01__T_A_site_active_infernos`: contribution `-0.004838`
- `lag_03__T_A_site_active_infernos`: contribution `-0.004693`
- `lag_07__T3__molly`: contribution `-0.002919`
- `lag_04__T4__molly`: contribution `-0.002835`

### tick `5434`, seconds `44.00`, LSTM delta `-0.1346`

Top all feature movements:
- `lag_01__T_place_MAIN`: contribution `-0.017839`
- `lag_04__CT_place_OUTSIDELONG`: contribution `-0.013833`
- `lag_00__CT_place_FOUNTAIN`: contribution `-0.007942`
- `lag_05__T_flash_duration_sum`: contribution `-0.006896`
- `lag_05__T2__flash_duration`: contribution `-0.005984`

Top utility-only movements:
- `lag_05__T_flash_duration_sum`: contribution `-0.006896`
- `lag_05__T2__flash_duration`: contribution `-0.005984`
- `lag_05__T1__flash_duration`: contribution `-0.005263`
- `lag_00__CT1__flash_duration`: contribution `+0.004211`
- `lag_05__T5__flash_duration`: contribution `-0.002732`

### tick `5786`, seconds `49.50`, LSTM delta `-0.0649`

Top all feature movements:
- `lag_07__CT_place_BRICKS`: contribution `-0.012595`
- `lag_00__T_kills_last_3s`: contribution `-0.005303`
- `lag_11__CT1__duck_amount`: contribution `-0.004220`
- `lag_00__T_damage_last_5s`: contribution `-0.004066`
- `lag_12__T_place_MAIN`: contribution `-0.003301`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `5274`, seconds `41.50`, LSTM delta `-0.0501`

Top all feature movements:
- `lag_00__T_flash_duration_sum`: contribution `-0.008162`
- `lag_00__T2__flash_duration`: contribution `-0.007121`
- `lag_11__CT1__duck_amount`: contribution `+0.006728`
- `lag_00__T1__flash_duration`: contribution `-0.006269`
- `lag_14__CT_place_OUTSIDELONG`: contribution `-0.004555`

Top utility-only movements:
- `lag_00__T_flash_duration_sum`: contribution `-0.008162`
- `lag_00__T2__flash_duration`: contribution `-0.007121`
- `lag_00__T1__flash_duration`: contribution `-0.006269`
- `lag_00__T5__flash_duration`: contribution `-0.003150`
- `lag_00__CT1__flash_duration`: contribution `-0.002923`

### tick `5658`, seconds `47.50`, LSTM delta `+0.0387`

Top all feature movements:
- `lag_00__T2__flash_duration`: contribution `+0.007121`
- `lag_03__CT_place_BRICKS`: contribution `+0.006404`
- `lag_00__T1__flash_duration`: contribution `+0.006269`
- `lag_00__T_flash_duration_sum`: contribution `+0.005970`
- `lag_02__T_place_FOUNTAIN`: contribution `+0.003177`

Top utility-only movements:
- `lag_00__T2__flash_duration`: contribution `+0.007121`
- `lag_00__T1__flash_duration`: contribution `+0.006269`
- `lag_00__T_flash_duration_sum`: contribution `+0.005970`
- `lag_02__T_A_site_active_infernos`: contribution `+0.002543`
- `lag_00__T_A_site_active_infernos`: contribution `+0.002212`
