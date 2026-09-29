"""Estatistika Deskribatzailearen modulua.

Machine Learning Utils liburutegiaren barruko modulua. Estatistika
deskribatzaileko oinarrizko funtzioak eskaintzen ditu, KANPOKO LIBURUTEGIRIK
ERABILI GABE. Bakarrik ``math`` moduluko ``ceil`` eta ``sqrt`` funtzioak
inportatzen dira.

Funtzioak:
    batez_bestekoa, mediana, pertzentila, bariantza,
    desbiderapen_tipikoa, laburpen_estatistikoa

Erabilera:
    >>> from machine_learning_utils import estatistika_deskribatzailea as ed
    >>> ed.batez_bestekoa([1, 2, 3, 4])
    2.5

Funtzio bakoitzaren dokumentazioa ikusteko: help(ed.mediana)
"""

from math import ceil, sqrt

__all__ = [
    "batez_bestekoa",
    "mediana",
    "pertzentila",
    "bariantza",
    "desbiderapen_tipikoa",
    "laburpen_estatistikoa",
]


# ---------------------------------------------------------------------------
# Barne-funtzio laguntzaileak (ez dira esportatzen)
# ---------------------------------------------------------------------------
def _balioztatu(datuak):
    """Datuak zerrenda ez-huts bat direla eta zenbakizkoak direla egiaztatzen du.

    Zerrenda (edo tupla) berri bat itzultzen du, jatorrizkoa aldatu gabe.
    Errore motak: TypeError (mota okerra) eta ValueError (zerrenda hutsa).
    """
    if not isinstance(datuak, (list, tuple)):
        raise TypeError("Datuak zerrenda edo tupla izan behar dira.")
    if len(datuak) == 0:
        raise ValueError("Datuen zerrenda ezin da hutsik egon.")
    for balioa in datuak:
        if isinstance(balioa, bool) or not isinstance(balioa, (int, float)):
            raise TypeError("Datu guztiek zenbakizkoak izan behar dute (int edo float).")
        if balioa != balioa:  # NaN detektatzeko
            raise ValueError("Datuek ezin dute NaN baliorik izan.")
    return list(datuak)


def _ordenatu(datuak):
    """Zerrenda goranzko ordenan itzultzen du (merge sort, O(n log n)).

    Ez da sorted() edo list.sort() erabiltzen: algoritmoa esplizituki
    inplementatuta dago.
    """
    if len(datuak) <= 1:
        return list(datuak)
    erdia = len(datuak) // 2
    ezkerra = _ordenatu(datuak[:erdia])
    eskuina = _ordenatu(datuak[erdia:])
    emaitza = []
    i = j = 0
    while i < len(ezkerra) and j < len(eskuina):
        if ezkerra[i] <= eskuina[j]:
            emaitza.append(ezkerra[i])
            i += 1
        else:
            emaitza.append(eskuina[j])
            j += 1
    emaitza.extend(ezkerra[i:])
    emaitza.extend(eskuina[j:])
    return emaitza


def _batura(datuak):
    """Balioen batura (for bidezko metatzailea, sum() erabili gabe)."""
    guztira = 0
    for balioa in datuak:
        guztira += balioa
    return guztira


# ---------------------------------------------------------------------------
# Funtzio publikoak
# ---------------------------------------------------------------------------
def batez_bestekoa(datuak):
    """Batez besteko aritmetikoa kalkulatzen du.

    Balio multzo bat batuz lortzen da, batugaien kopuru osoarekin zatituta:
        media = (x1 + x2 + ... + xn) / n

    Parametroak:
        datuak (list | tuple): zenbakizko balioen zerrenda ez-hutsa.

    Itzultzen du:
        float: datuen batez besteko aritmetikoa.

    Erroreak:
        TypeError: datuak ez badira zerrenda/tupla edo ez badira zenbakizkoak.
        ValueError: zerrenda hutsik badago edo NaN baliorik badago.

    Adibidea:
        >>> batez_bestekoa([2, 4, 6, 8])
        5.0
    """
    balioak = _balioztatu(datuak)
    return _batura(balioak) / len(balioak)


def mediana(datuak):
    """Datuen mediana kalkulatzen du.

    Ordenatutako datu-multzo batean erdiko posizioko aldagaiaren balioa
    adierazten du; datuen erdiak balio horretatik behera geratzen dira.
    Datu kopurua bikoitia bada, erdiko bi balioen batez bestekoa da.

    Parametroak:
        datuak (list | tuple): zenbakizko balioen zerrenda ez-hutsa
            (ez da beharrezkoa ordenatuta egotea).

    Itzultzen du:
        float | int: mediana.

    Erroreak:
        TypeError: datuak ez badira zerrenda/tupla edo ez badira zenbakizkoak.
        ValueError: zerrenda hutsik badago edo NaN baliorik badago.

    Adibideak:
        >>> mediana([7, 1, 3])
        3
        >>> mediana([1, 2, 3, 4])
        2.5
    """
    ordenatuta = _ordenatu(_balioztatu(datuak))
    n = len(ordenatuta)
    erdia = n // 2
    if n % 2 == 1:
        return ordenatuta[erdia]
    return (ordenatuta[erdia - 1] + ordenatuta[erdia]) / 2


def pertzentila(datuak, p):
    """p pertzentila kalkulatzen du (hurbilen dagoen postuaren metodoa).

    Pertzentila aldagaiaren balio bat da, eta haren azpitik behaketen
    ehuneko jakin bat (p %) dago talde batean. Metodoa ("nearest-rank"):
        postua = ceil(p / 100 * n)      (gutxienez 1)
        pertzentila = ordenatutako datuen [postua]-garren balioa
    Beraz, itzulitako balioa beti da datuetako bat.

    Oharra: p = 50 denean eta n bikoitia denean, emaitza ez dator bat
    ``mediana`` funtzioarekin (hark erdiko bi balioen batez bestekoa
    hartzen du; hemen, berriz, beheko erdiko balioa).

    Parametroak:
        datuak (list | tuple): zenbakizko balioen zerrenda ez-hutsa.
        p (int | float): pertzentila, 0 eta 100 artean (biak barne).

    Itzultzen du:
        float | int: p pertzentilaren balioa.

    Erroreak:
        TypeError: datuak edo p mota okerrekoak badira.
        ValueError: p 0-100 tartetik kanpo badago, zerrenda hutsik badago
            edo NaN baliorik badago.

    Adibidea:
        >>> pertzentila([15, 20, 35, 40, 50], 40)
        20
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise TypeError("p zenbaki bat izan behar da.")
    if not 0 <= p <= 100:
        raise ValueError("p 0 eta 100 artean egon behar da.")
    ordenatuta = _ordenatu(_balioztatu(datuak))
    postua = ceil(p / 100 * len(ordenatuta))
    if postua < 1:
        postua = 1
    return ordenatuta[postua - 1]


def bariantza(datuak, lagina=False):
    """Bariantza kalkulatzen du.

    Aldagai bakoitzak batez besteko aritmetikoarekiko duen distantzia
    karratua kalkulatzen da eta, ondoren, distantzia horien batez bestekoa:
        bariantza = sum((xi - media) ** 2) / n

    Lagin baten bariantza nahi bada (``lagina=True``), n-ren ordez
    (n - 1) erabiltzen da (Besselen zuzenketa).

    Parametroak:
        datuak (list | tuple): zenbakizko balioen zerrenda ez-hutsa.
        lagina (bool): True bada, lagin-bariantza (n - 1). Lehenetsia:
            False (populazio-bariantza, n).

    Itzultzen du:
        float: bariantza.

    Erroreak:
        TypeError: datuak ez badira zerrenda/tupla edo ez badira zenbakizkoak.
        ValueError: zerrenda hutsik badago, NaN baliorik badago, edo
            lagina=True eta datu bakarra badago.

    Adibidea:
        >>> bariantza([2, 4, 4, 4, 5, 5, 7, 9])
        4.0
    """
    balioak = _balioztatu(datuak)
    n = len(balioak)
    if lagina and n < 2:
        raise ValueError("Lagin-bariantzarako gutxienez 2 datu behar dira.")
    media = _batura(balioak) / n
    karratuak = [(x - media) ** 2 for x in balioak]
    zatitzailea = n - 1 if lagina else n
    return _batura(karratuak) / zatitzailea


def desbiderapen_tipikoa(datuak, lagina=False):
    """Desbiderapen tipikoa kalkulatzen du.

    Bariantzaren erro karratu positiboa da, eta datuak batez bestekoaren
    inguruan zenbat sakabanatuta dauden neurtzen du (datuen unitate berean):
        desbiderapena = sqrt(bariantza)

    Parametroak:
        datuak (list | tuple): zenbakizko balioen zerrenda ez-hutsa.
        lagina (bool): True bada, lagin-desbiderapena (n - 1). Lehenetsia:
            False (populazio-desbiderapena).

    Itzultzen du:
        float: desbiderapen tipikoa.

    Erroreak:
        TypeError: datuak ez badira zerrenda/tupla edo ez badira zenbakizkoak.
        ValueError: zerrenda hutsik badago, NaN baliorik badago, edo
            lagina=True eta datu bakarra badago.

    Adibidea:
        >>> desbiderapen_tipikoa([2, 4, 4, 4, 5, 5, 7, 9])
        2.0
    """
    return sqrt(bariantza(datuak, lagina))


def laburpen_estatistikoa(datuak):  
    """Datu-zerrenda baten laburpen estatistikoa kalkulatzen du.

    Batez besteko aritmetikoa, mediana, balio minimoa eta maximoa,
    25, 50 eta 75 pertzentilak, bariantza eta desbiderapen tipikoa
    kalkulatzen ditu, eta hiztegi batean itzultzen ditu.

    Hiztegiaren gakoak:
        "batez_bestekoa", "mediana", "minimoa", "maximoa",
        "p25", "p50", "p75", "bariantza", "desbiderapen_tipikoa"

    Parametroak:
        datuak (list | tuple): zenbakizko balioen zerrenda ez-hutsa.

    Itzultzen du:
        dict: gako bakoitzeko balio estatistikoa.

    Erroreak:
        TypeError: datuak ez badira zerrenda/tupla edo ez badira zenbakizkoak.
        ValueError: zerrenda hutsik badago edo NaN baliorik badago.

    Adibidea:
        >>> r = laburpen_estatistikoa([1, 2, 3, 4, 5])
        >>> r["batez_bestekoa"], r["mediana"], r["p75"]
        (3.0, 3, 4)
    """
    balioak = _balioztatu(datuak)
    ordenatuta = _ordenatu(balioak)
    return {
        "batez_bestekoa": batez_bestekoa(balioak),
        "mediana": mediana(balioak),
        "minimoa": ordenatuta[0],
        "maximoa": ordenatuta[-1],
        "p25": pertzentila(balioak, 25),
        "p50": pertzentila(balioak, 50),
        "p75": pertzentila(balioak, 75),
        "bariantza": bariantza(balioak),
        "desbiderapen_tipikoa": desbiderapen_tipikoa(balioak),
    }
