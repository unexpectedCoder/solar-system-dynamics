from typing import Dict
from matplotlib.colors import Normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lina

from solsys_data import AU, G_CONST, M_SUN
import solsys_data as solsysd


def solve(m, r0, v0, time: float, dt: float):
    t = 0.
    m = np.asarray(m)
    r = np.asarray(r0)
    v = np.asarray(v0)
    a = _calc_accel(m, r)
    res: Dict[str, list] = {
        "t": [t], "r": [r], "v": [v], "a": [a]
    }
    pbar = tqdm(
        desc="Прогресс",
        total=int(time/dt),
        colour="green"
    )
    while t < time:
        v = v + a*dt
        r = r + v*dt
        a = _calc_accel(m, r)
        t += dt
        if pbar.n % 100 == 0:
            res["r"].append(r)
            res["v"].append(v)
            res["a"].append(a)
            res["t"].append(t)
        pbar.update(1)
    pbar.close()
    return res


def _calc_accel(m: np.ndarray,
                r: np.ndarray):
    # Тяготение от неподвижного Солнца
    r_mag3 = lina.norm(r, axis=1)**3
    r_ratio = \
        (r.flatten() / r_mag3.repeat(3)).reshape([m.size, 3])
    # приводит к такому ускорению планет
    a = -G_CONST * M_SUN * r_ratio
    # Взаимодействие планет
    f = np.zeros_like(r)
    for i in range(m.size - 1):
        for j in range(i + 1, m.size):
            r_ij = r[j] - r[i]
            r3 = lina.norm(r_ij)**3
            f_ij = G_CONST * m[i] * m[j] * r_ij / r3
            f[i] += f_ij
            f[j] -= f_ij
    a += (
        f.flatten() / m.repeat(3)
    ).reshape([m.size, 3])
    return a


def plot_sol_sys(m, data: Dict[str, list], figax=None):
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    names = solsysd.planets_name()
    ax.plot(
        [0], [0],
        label="Солнце", ls="", c="k", marker="*"
    )
    norm = Normalize(np.min(m), np.max(m))
    for name, mi, ri in zip(names, m, data["r"]):
        ax.scatter(
            ri[0]/AU, ri[1]/AU,
            c=mi, cmap="inferno", norm=norm,
            label=name
        )
    ax.set(
        xlabel="$x$, а. е.",
        ylabel="$y$, а. е.",
        aspect="equal"
    )
    ax.legend(frameon=False)
    return fig, ax


def plot_sol_sys_in_motion(data: Dict[str, list],
                           figax=None):
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    ax.plot(
        [0], [0],
        label="Солнце", ls="", c="k", marker="*"
    )
    r = np.array(data["r"])
    names = solsysd.planets_name()
    for i, name in enumerate(names):
        ax.plot(
            r[:, i, 0]/AU, r[:, i, 1]/AU,
            label=name
        )
    norm = Normalize(np.min(m), np.max(m))
    for name, mi, ri in zip(names, m, data["r"][-1]):
        ax.scatter(
            ri[0]/AU, ri[1]/AU,
            c=mi, cmap="inferno", norm=norm
        )
    ax.set(
        xlabel="$x$, а. е.",
        ylabel="$y$, а. е.",
        aspect="equal"
    )
    ax.legend(frameon=False)
    return fig, ax


if __name__ == "__main__":
    from datetime import datetime, timedelta

    when1 = datetime.now()
    when2 = when1 + timedelta(hours=1)
    m = solsysd.planets_mass()
    r = solsysd.planets_position(when1)
    v = solsysd.planets_velocity(when1, when2)
    with plt.style.context("fast"):
        plot_sol_sys(m, {"r": r})
        plt.show()
    
    res = solve(
        m,
        r,
        v,
        timedelta(days=365).total_seconds(),
        timedelta(minutes=5).total_seconds()
    )
    with plt.style.context("fast"):
        plot_sol_sys_in_motion(res)
        plt.show()
