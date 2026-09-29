"""Machine Learning Utils liburutegia.

IA sailaren kode berrerabilgarrien bilduma. Moduluak:
    estatistika_deskribatzailea -> estatistika deskribatzaileko funtzioak
                                   (kanpoko liburutegirik gabe).
"""

from . import estatistika_deskribatzailea
from .estatistika_deskribatzailea import (
    batez_bestekoa,
    mediana,
    pertzentila,
    bariantza,
    desbiderapen_tipikoa,
    laburpen_estatistikoa,
)

__version__ = "1.0.0"
__all__ = [
    "estatistika_deskribatzailea",
    "batez_bestekoa",
    "mediana",
    "pertzentila",
    "bariantza",
    "desbiderapen_tipikoa",
    "laburpen_estatistikoa",
]
