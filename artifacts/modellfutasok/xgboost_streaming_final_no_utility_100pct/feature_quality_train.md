# Feature Quality Report

- run: `xgboost_streaming_final_no_utility_100pct`
- split scanned: `train`
- feature count in metrics: `412`
- scanned feature count: `412`
- forbidden hits: `0`
- leak-suspicious hits: `0`
- constant features: `15`
- rare nonzero features: `19`

## Block Counts

- `position`: `249`
- `state`: `41`
- `player_slot_numeric`: `30`
- `objective`: `28`
- `utility`: `22`
- `combat`: `18`
- `economy`: `12`
- `other`: `10`
- `time`: `2`

## Forbidden Hits

- none

## Leak Suspicious Hits

- none

## Constant Features

- `CT1__has_bomb`
- `CT2__has_bomb`
- `CT3__has_bomb`
- `CT4__has_bomb`
- `CT5__has_bomb`
- `CT_bomb_carrier_alive`
- `CT_bomb_zone_count`
- `T1__has_defuser`
- `T2__has_defuser`
- `T3__has_defuser`
- `T4__has_defuser`
- `T5__has_defuser`
- `T_defuser_count`
- `T_defusing_count`
- `T_place_UNKNOWN`

## Rare Nonzero Features

- `CT_place_DECK` nonzero_ratio=`0.000056` block=`position`
- `CT_place_DUMPSTER` nonzero_ratio=`0.000354` block=`position`
- `CT_place_KITCHEN` nonzero_ratio=`0.000360` block=`utility`
- `CT_place_PIPE` nonzero_ratio=`0.000383` block=`position`
- `CT_place_PLAYGROUND` nonzero_ratio=`0.000371` block=`position`
- `CT_place_SIDE` nonzero_ratio=`0.000943` block=`position`
- `CT_place_SILO` nonzero_ratio=`0.000184` block=`position`
- `CT_place_STREET` nonzero_ratio=`0.000898` block=`position`
- `CT_place_UPSTAIRS` nonzero_ratio=`0.000562` block=`position`
- `T_place_BACKOFA` nonzero_ratio=`0.000493` block=`position`
- `T_place_BRICKS` nonzero_ratio=`0.000293` block=`position`
- `T_place_CRANE` nonzero_ratio=`0.000245` block=`position`
- `T_place_CTSIDEUPPER` nonzero_ratio=`0.000429` block=`position`
- `T_place_ENTRANCE` nonzero_ratio=`0.000834` block=`position`
- `T_place_HUTROOF` nonzero_ratio=`0.000903` block=`position`
- `T_place_KITCHEN` nonzero_ratio=`0.000769` block=`utility`
- `T_place_LOCKERROOM` nonzero_ratio=`0.000656` block=`position`
- `T_place_SCAFFOLDING` nonzero_ratio=`0.000831` block=`position`
- `T_place_STORAGEROOM` nonzero_ratio=`0.000290` block=`position`

## Rarest Utility Features

- `CT_place_KITCHEN` nonzero_ratio=`0.000360` unique_count=`3.0`
- `T_place_KITCHEN` nonzero_ratio=`0.000769` unique_count=`4.0`
- `T_place_HELL` nonzero_ratio=`0.001488` unique_count=`5.0`
- `T_place_HEAVEN` nonzero_ratio=`0.004723` unique_count=`5.0`
- `CT_place_HELL` nonzero_ratio=`0.014822` unique_count=`6.0`
- `CT_place_HEAVEN` nonzero_ratio=`0.027557` unique_count=`5.0`
- `T_flashed_players` nonzero_ratio=`0.089157` unique_count=`6.0`
- `CT_flashed_players` nonzero_ratio=`0.090036` unique_count=`6.0`
- `CT3__has_helmet` nonzero_ratio=`0.435050` unique_count=`2.0`
- `CT2__has_helmet` nonzero_ratio=`0.448689` unique_count=`2.0`
- `CT5__has_helmet` nonzero_ratio=`0.450180` unique_count=`2.0`
- `CT4__has_helmet` nonzero_ratio=`0.453405` unique_count=`2.0`
- `CT1__has_helmet` nonzero_ratio=`0.468908` unique_count=`2.0`
- `T1__has_helmet` nonzero_ratio=`0.640092` unique_count=`2.0`
- `T4__has_helmet` nonzero_ratio=`0.643343` unique_count=`2.0`
- `T2__has_helmet` nonzero_ratio=`0.644830` unique_count=`2.0`
- `T5__has_helmet` nonzero_ratio=`0.646231` unique_count=`2.0`
- `T3__has_helmet` nonzero_ratio=`0.648602` unique_count=`2.0`
- `CT_helmet_count` nonzero_ratio=`0.756818` unique_count=`6.0`
- `T_helmet_count` nonzero_ratio=`0.856759` unique_count=`6.0`
- `T_macro_OTHER` nonzero_ratio=`0.910005` unique_count=`6.0`
- `CT_macro_OTHER` nonzero_ratio=`0.937181` unique_count=`6.0`
