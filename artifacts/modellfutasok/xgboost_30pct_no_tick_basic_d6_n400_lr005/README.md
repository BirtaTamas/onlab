# XGBoost 30pct no tick basic d6 n400 lr005

Ez a futas sikeresen lefutott.

## Beallitas

- adatforras: `processed_full`
- output mappa: `artifacts/modellfutasok/xgboost_30pct_no_tick_basic_d6_n400_lr005`
- minta: `sample_csv_ratio=0.3`
- ritkitas: `row_stride=2`
- XGBoost mod: `--use-library-defaults`
- kezzel megadott fo parameterek:
  - `max_depth=6`
  - `n_estimators=400`
  - `learning_rate=0.05`

## Feature setup

- `tick` nincs benne
- kategoriás feature-ok nincsenek bekapcsolva
- utility feature-ok bent maradtak
- eros nem-utility feature-ok is bent maradtak
- nincs extra feature-szures

## Test eredmenyek

- accuracy: `0.7490`
- precision: `0.7479`
- recall: `0.7803`
- f1_score: `0.7637`
- brier_score: `0.1601`
- logloss: `0.4734`
- roc_auc: `0.8445`

## Accuracy

Ertek: `0.7490`

Mit mer:

- az osszes predikciobol mekkora arany lett helyes

Mikor jo:

- ha egy gyors, konnyen ertheto osszkep kell
- ha a ket osztaly nincs nagyon elcsuszva egymastol

Mikor nem jo onmagaban:

- ha fontos, hogy milyen tipusuak a hibak
- ha az egyik osztaly joval gyakoribb, mert ilyenkor felrevezeto lehet

Miert lehet jo ez az eredmeny:

- kozel 75%-os helyes talalati arany egy jo baseline
- a fo numerikus, eros feature-ok bent maradtak

Miert johetett ki pont ilyen:

- a kisebb mintavetel miatt kicsit zajosabb lehet a becsles
- a `tick` hianya nem okozott nagy romlast

## Precision

Ertek: `0.7479`

Mit mer:

- amikor a modell CT wint mond, azok kozul mennyi tenyleg CT win

Mikor jo:

- ha fontos, hogy a pozitiv predikcio megbizhato legyen
- ha zavar a sok false positive

Mikor nem jo onmagaban:

- ha kozben sok valodi pozitiv esetet hagy ki a modell

Miert lehet jo ez az eredmeny:

- a pozitivnak jelolt esetek nagyjabol haromnegyede helyes
- ez egy stabil, vallalhato baseline-szint

Miert johetett ki pont ilyen:

- a modell inkabb picit nyitottabb a pozitiv predikciokra
- emiatt a precision jo marad, de nem lesz tul magas

## Recall

Ertek: `0.7803`

Mit mer:

- a valodi CT win esetekbol mennyit talal meg a modell

Mikor jo:

- ha fontos, hogy minel kevesebb pozitiv eset maradjon ki

Mikor nem jo onmagaban:

- ha kozben tul sok teves pozitiv esetet gyart

Miert lehet jo ez az eredmeny:

- a modell a pozitiv esetek nagy reszet elkapja
- ez azt sugallja, hogy nem tul konzervativ

Miert johetett ki pont ilyen:

- a 0.5-os thresholdnal a modell inkabb recall-felen all
- a bent hagyott eros feature-ok sok pozitiv mintat jol felismerhetove tesznek

## F1 score

Ertek: `0.7637`

Mit mer:

- a precision es recall kozos, kiegyensulyozott mutatoja

Mikor jo:

- ha nem egyetlen metrikat akarsz nezni a pozitiv osztalyra
- ha egyszerre fontos a precision es a recall

Mikor nem jo onmagaban:

- ha a probability becsles minosege az erdekes, nem csak a cimke

Miert lehet jo ez az eredmeny:

- jo kompromisszum van a precision es recall kozott
- egyik sem esik be latvanyosan

Miert johetett ki pont ilyen:

- a precision es recall eleg kozel vannak egymashoz
- emiatt az F1 is stabilan eros marad

## ROC AUC

Ertek: `0.8445`

Mit mer:

- mennyire jol rangsorolja a modell a pozitiv es negativ eseteket a valoszinusegek alapjan

Mikor jo:

- ha a win probability sorrendje is fontos
- ha nem csak egy fix threshold erdekel

Mikor nem jo onmagaban:

- nem mondja meg, hogy a probability-k jol vannak-e kalibralva
- lehet jo AUC mellett is gyengebb thresholdos osztalyozas

Miert lehet jo ez az eredmeny:

- `0.8445` mar jo szeparacios kepesseget jelent
- a modell altalaban a pozitiv eseteket magasabb valoszinusegre rakja

Miert johetett ki pont ilyen:

- az XGBoost jol kihasznalja a nemlinearis mintazatokat
- a fontos allapotjelzo feature-ok eleg sok informaciot hordoznak

## Logloss

Ertek: `0.4734`

Mit mer:

- mennyire jok a prediktalt valoszinusegek
- a magabiztos tevedeseket jobban bunteti

Mikor jo:

- ha nemcsak osztalyozni, hanem valoszinuseget is becsulni akarunk

Mikor nem jo onmagaban:

- kevesbe intuitiv, mint az accuracy
- inkabb osszehasonlitasra jo kulonbozo futasok kozt

Miert lehet jo ez az eredmeny:

- a probability outputok hasznalhatoak
- nincs sok nagyon rossz, magabiztos tipp

Miert johetett ki pont ilyen:

- a modell a legtobb savban ertelmes bizonytalansagot ad
- a kozepso tartomany kisebb kalibracios hibai rontanak rajta valamennyit

## Brier score

Ertek: `0.1601`

Mit mer:

- a prediktalt valoszinuseg es a tenyleges kimenet negyzetes eltereset

Mikor jo:

- ha a valoszinusegi becslesek minosege fontos
- ha a kalibraciot is figyelni akarod

Mikor nem jo onmagaban:

- nem mondja meg kulon, hogy a hiba szeparacios vagy kalibracios eredetu

Miert lehet jo ez az eredmeny:

- a modell outputja eleg stabil
- a probability-k nem tunnek teljesen tulbiztosnak vagy osszevisszanak

Miert johetett ki pont ilyen:

- az alacsony es magas valoszinusegi savokban jo az illeszkedes
- a kozepso tartomanyban van kisebb bizonytalansag

## Confusion matrix

`[[39550, 15762], [13162, 46752]]`

Felbontva:

- TN: `39550`
- FP: `15762`
- FN: `13162`
- TP: `46752`

Mit mutat:

- `TN`: helyes negativ predikciok
- `FP`: teves pozitiv predikciok
- `FN`: kihagyott pozitiv esetek
- `TP`: helyes pozitiv predikciok

Mikor jo:

- ha latni akarjuk, milyen iranyban hibazik a modell

Mikor nem jo onmagaban:

- thresholdfuggo, most a `0.5`-os kuszobhoz tartozik
- nem mutatja a valoszinusegek minoseget

Miert lehet jo ez az eredmeny:

- a modell sok pozitiv esetet jol elkap
- a `TP` magasabb, mint a `FN`, ami osszhangban van a jo recallal

Miert johetett ki pont ilyen:

- a modell inkabb pozitivabb iranyba hajlik
- emiatt a `FP` sem alacsony, vagyis a precision-re van egy termeszetes nyomas

## Calibration curve

Mit mutat:

- hogy a prediktalt valoszinusegek mennyire fedik a tenyleges pozitiv aranyokat

Mikor jo:

- ha a `0.7` vagy `0.9` jellegu becsleseket tenyleges valoszinusegkent akarjuk kezelni

Mikor nem jo onmagaban:

- a bineles miatt leegyszerusit
- jo kalibracio mellett is lehet gyenge a szeparacio

Miert lehet jo ez az eredmeny:

- az alacsony es magas savokban eleg jo az egyezes
- a modell probability outputja emiatt ertelmezheto

Miert johetett ki pont ilyen:

- kisebb mintan futottunk, ez zajosabba teszi a kozeptartomanyt
- a modell eros a szeparacioban, de nem tokeletesen kalibralt minden binben

## Gyors ertekeles

Ez a futas egy jo baseline lett kisebb mintan. Az `accuracy` kozel `0.75`, az `f1_score` `0.76` korul van, a `roc_auc` pedig `0.8445`, ami azt mutatja, hogy a modell jol rangsorolja a CT win valoszinuseget. A `recall` magasabb, mint a `precision`, vagyis a modell valamivel gyakrabban jelol pozitivnak, de cserébe tobb valodi pozitiv esetet is megtalal.

A kalibracio osszkepre jonak tunik: a kozepso savokban van kisebb elteres, de a magas valoszinusegu bin-eknel a predikciok egeszen kozel vannak a tenyleges pozitiv aranyhoz. A `brier_score` es a `logloss` alapjan a valoszinusegi becslesek is hasznalhatok, nemcsak a `0.5`-os osztalyozas.

## Fajlok

- `metrics.json`: teljes metrika-osszegzes
- `split_manifest.csv`: teljes split lista
- `sampled_split_manifest.csv`: a tenylegesen hasznalt mintazott CSV-k
- `xgboost_model.json`: a mentett modell
- `train_confusion_matrix.png`, `valid_confusion_matrix.png`, `test_confusion_matrix.png`
- `train_calibration_curve.png`, `valid_calibration_curve.png`, `test_calibration_curve.png`
