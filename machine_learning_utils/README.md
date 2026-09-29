# Machine Learning Utils

Liburutegi berrerabilgarria IA sailerako.

## Egitura
machine_learning_utils/
    __init__.py
    estatistika_deskribatzailea.py
tests/

## Erabilera
    from machine_learning_utils import estatistika_deskribatzailea as ed
    ed.laburpen_estatistikoa([4, 8, 15, 16, 23, 42])
    help(ed.mediana)

## Probak
    python -m unittest discover -s tests -t .

Kanpoko libururik ez da erabiltzen (soilik math.ceil eta math.sqrt).
