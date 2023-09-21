from datetime import datetime
import numpy as np
import solarsystem as solsys


G_CONST = 6.6743e-11
M_SUN = 1.988e30
AU = 1.495978707e11


def planets_position(when: datetime):
    helio = solsys.Heliocentric(
        *when.timetuple()[:5], view="rectangular"
    )
    pos = helio.planets()
    return AU * np.array([
        pos["Mercury"],
        pos["Venus"],
        pos["Earth"],
        pos["Mars"],
        pos["Jupiter"],
        pos["Saturn"],
        pos["Uranus"],
        pos["Neptune"],
        pos["Pluto"]
    ])


def planets_velocity(when1: datetime,
                     when2: datetime):
    pos1 = planets_position(when1)
    pos2 = planets_position(when2)
    dr = pos2 - pos1
    dt = when2 - when1
    return dr / dt.total_seconds()


def planets_mass():
    return np.array([
        0.32868e24,
        4.8685e24,
        5.9736e24,
        0.64185e24,
        1898.68e24,
        568.46e24,
        86.832e24,
        102.43e24,
        0.013105e24
    ])


def planets_name():
    return (
        "Меркурий",
        "Венера",
        "Земля",
        "Марс",
        "Юпитер",
        "Сатурн",
        "Уран",
        "Нептун",
        "Плутон"
    )
