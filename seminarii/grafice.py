from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
import seaborn as sns


def plot_varianta(valori_proprii_sortate: np.ndarray, procent_minimal=80, scal=True):
    # Logica pentru Scree Plot rămâne aceeași, asigură-te că salvezi în folderul corect
    nr_componente = len(valori_proprii_sortate)
    axe_x_componente = np.arange(1, nr_componente + 1)
    figura = plt.figure(figsize=(8, 5))
    axa_grafic = figura.add_subplot(1, 1, 1)

    axa_grafic.set_title("Plot varianta componente", fontdict={"color": "b", "fontsize": 16})
    axa_grafic.plot(axe_x_componente, valori_proprii_sortate)
    axa_grafic.set_xlabel("Componente/Factori", fontsize=12)
    axa_grafic.set_ylabel("Varianta (Eigenvalues)", fontsize=12)
    axa_grafic.set_xticks(axe_x_componente)
    axa_grafic.scatter(axe_x_componente, valori_proprii_sortate, c="r")

    k1 = None
    if scal:
        axa_grafic.axhline(1, c="g", label="Criteriul Kaiser")
        k1 = len(np.where(valori_proprii_sortate > 1)[0])

    procent_cumulat = np.cumsum(valori_proprii_sortate * 100 / np.sum(valori_proprii_sortate))
    index_acoperire = np.where(procent_cumulat > procent_minimal)[0][0]
    k2 = index_acoperire + 1
    axa_grafic.axhline(valori_proprii_sortate[index_acoperire], c="m", label=f"Acoperire {procent_minimal}%")

    # Criteriul Cattell (Cotul)
    dif1 = valori_proprii_sortate[:-1] - valori_proprii_sortate[1:]
    dif2 = dif1[:-1] - dif1[1:]
    k3 = None
    if any(dif2 < 0):
        k3 = np.where(dif2 < 0)[0][0] + 2
        axa_grafic.axhline(valori_proprii_sortate[k3 - 1], c="c", label="Criteriul Cattell")

    axa_grafic.legend()
    plt.savefig("graphics/PlotVarianta.png")
    return k1, k2, k3


def corelograma(dataframe_date: pd.DataFrame, titlu="Corelograma", vmin=-1, vmax=1, cmap="RdYlBu", annot=True):
    """
    Versiune îmbunătățită care acceptă vmax variabil (ex: 100 pentru contribuții).
    """
    figura = plt.figure(figsize=(9, 8))
    axa_grafic = figura.add_subplot(1, 1, 1)
    axa_grafic.set_title(titlu, fontdict={"color": "b", "fontsize": 16})

    # Folosim vmax ca parametru pentru a suporta Contribuții (0-100) și Corelații (-1, 1)
    heatmap(dataframe_date, vmin=vmin, vmax=vmax, cmap=cmap, annot=annot, ax=axa_grafic)
    plt.savefig("graphics/" + titlu + ".png")


def plot_scoruri_corelatii(t: pd.DataFrame, varx="C1", vary="C2", titlu="Plot", etichete=None, corelatii=False):
    """
    Funcție unificată pentru scoruri (ACP) și cercul corelațiilor (Interpretare).
    """
    figura = plt.figure(figsize=(8, 8))
    ax = figura.add_subplot(1, 1, 1, aspect=1)  # aspect=1 pentru cerc perfect
    ax.set_title(titlu, fontdict={"color": "b", "fontsize": 16})
    ax.set_xlabel(varx)
    ax.set_ylabel(vary)

    if corelatii:
        # Cercul unitate și cercul de 70% (semnificație statistică)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='b', label="Cercul unitate")
        ax.plot(0.7 * np.cos(theta), 0.7 * np.sin(theta), color='g', linestyle='--', label="Prag 70%")
        ax.legend()

    ax.scatter(t[varx], t[vary], c="r", alpha=0.6)
    ax.axvline(0, c="k", linestyle=':')
    ax.axhline(0, c="k", linestyle=':')

    if etichete is not None:
        for i in range(len(t)):
            ax.text(t[varx].iloc[i], t[vary].iloc[i], etichete[i], fontsize=8)

    plt.savefig(f"graphics/{titlu}_{varx}_{vary}.png")


def cercul_corelatiilor(df_corelatii, varx="C1", vary="C2", titlu="Cercul Corelatiilor"):
    """
    Specializată pentru AF/ACP interpretativ, adaugă săgeți (vectori).
    """
    f = plt.figure(figsize=(7, 7))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu, color='b', fontsize=16)

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='b')
    ax.axhline(0, color='k');
    ax.axvline(0, color='k')

    for i in range(len(df_corelatii)):
        # Adăugăm vectorul de la origine la punctul corelației
        ax.arrow(0, 0, df_corelatii[varx].iloc[i], df_corelatii[vary].iloc[i],
                 color='r', alpha=0.5, head_width=0.03)
        ax.annotate(df_corelatii.index[i], (df_corelatii[varx].iloc[i], df_corelatii[vary].iloc[i]))

    plt.savefig(f"graphics/{titlu}.png")


def show():
    plt.show()


def plot_ierarhie(h: np.ndarray, etichete=None, color_threshold=0, titlu="Plot Ierarhie"):
    """
    Generează dendrograma pentru analiza de cluster ierarhic.
    """
    f = plt.figure(titlu, figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})

    # color_threshold determină unde se schimbă culoarea ramurilor (tăierea ierarhiei)
    dendrogram(h, color_threshold=color_threshold, labels=etichete, ax=ax)

    # Dacă avem un prag de tăiere, tragem o linie orizontală roșie
    if color_threshold != 0:
        ax.axhline(color_threshold, c="r", linestyle="--")

    plt.savefig(f"graphics/{titlu}.png")


def plot_harta(gdf, coloana_legatura, df_date, variabila, titlu="Harta"):
    """
    Vizualizează distribuția spațială a scorurilor factoriale sau a clusterelor.
    """
    # Combinăm datele geografice cu rezultatele analizei
    temp = gdf.merge(df_date[[variabila]], left_on=coloana_legatura, right_index=True)

    f = plt.figure(figsize=(10, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu + " - " + variabila, fontdict={"fontsize": 16})

    # Redăm harta folosind o paletă de culori divergentă
    temp.plot(column=variabila, legend=True, ax=ax, cmap="RdYlGn")

    plt.savefig(f"graphics/{titlu}_{variabila}.png")


def f_distributii(t_date, nume_variabila, nume_tinta, clase):
    """
    Genereaza ploturi de distributie (KDE) pentru o variabila, colorate dupa clasa.
    """
    figura = plt.figure(figsize=(8, 5))
    ax = figura.add_subplot(1, 1, 1)
    ax.set_title(f"Distributia {nume_variabila} pe clase", fontsize=14, color='b')

    # Desenam distributia pentru fiecare clasa
    for clasa in clase:
        # Selectam doar randurile care apartin clasei curente
        subset = t_date[t_date[nume_tinta] == clasa]
        sns.kdeplot(subset[nume_variabila], label=str(clasa), ax=ax, fill=True, alpha=0.3)

    ax.legend()
    plt.savefig(f"graphics/Distributie_{nume_variabila}.png")


def f_scatter(t_z, t_gz, etichete_clase, nume_clase):
    """
    Scatter plot pentru primele doua axe discriminante (Z1, Z2).
    t_z: Scorurile discriminante (instantele)
    t_gz: Centrele de greutate ale claselor
    etichete_clase: Vectorul cu clasa reala a fiecarei instante
    """
    figura = plt.figure(figsize=(9, 6))
    ax = figura.add_subplot(1, 1, 1)
    ax.set_title("Plot Scoruri Discriminante (Z1 vs Z2)", fontsize=16, color='b')
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")

    # Daca avem doar o singura axa discriminanta (2 clase), facem un plot 1D sau histograma
    if t_z.shape[1] == 1:
        sns.scatterplot(x=t_z.iloc[:, 0], y=np.zeros(len(t_z)), hue=etichete_clase, style=etichete_clase, ax=ax, s=50)
        # Plotam si centrii
        ax.scatter(t_gz.iloc[:, 0], np.zeros(len(t_gz)), c="black", marker="X", s=200, label="Centroizi")
    else:
        # Plotam instantele
        sns.scatterplot(x=t_z.iloc[:, 0], y=t_z.iloc[:, 1], hue=etichete_clase, style=etichete_clase, ax=ax, s=50)

        # Plotam centrii de greutate
        ax.scatter(t_gz.iloc[:, 0], t_gz.iloc[:, 1], c="black", marker="X", s=200, label="Centroizi")

        # Etichetam centrii
        for i, txt in enumerate(nume_clase):
            ax.text(t_gz.iloc[i, 0], t_gz.iloc[i, 1], str(txt), fontsize=12, fontweight='bold')

    plt.legend()
    plt.savefig("graphics/Plot_Discriminant_Z.png")