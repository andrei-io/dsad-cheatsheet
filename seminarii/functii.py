import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import shapiro, kstest, norm, chi2
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

def calcul_metrici(y_true, y_pred, clase):
    """
    Calculeaza matricea de confuzie, acuratetea globala si indexul Cohen-Kappa.
    """
    # 1. Matricea de confuzie
    cm = confusion_matrix(y_true, y_pred, labels=clase)
    df_cm = pd.DataFrame(cm, index=clase, columns=clase)
    df_cm.index.name = "Real"
    df_cm.columns.name = "Predictie"

    # 2. Acuratetea si Kappa
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    df_metrics = pd.DataFrame({
        "Acuratete": [acc],
        "Kappa": [kappa]
    }, index=["Valori"])

    return df_cm, df_metrics

def nan_replace(matrice_date: np.ndarray):
    # Gaseste toate aparitiile valorilor NaN - returneaza o matrice bitmask
    este_nan = np.isnan(matrice_date)

    # Obtine indicii (rand, coloana) pentru fiecare valoare NaN
    indici_nan = np.where(este_nan)

    # Calculeaza media pentru fiecare coloana care contine cel putin un NaN
    medii_coloane = np.nanmean(matrice_date[:, indici_nan[1]], axis=0)

    # Inlocuieste valorile NaN (identificate prin 'indici_nan')
    # cu media corespunzatoare a coloanei lor
    matrice_date[indici_nan] = medii_coloane


def teste_concordanta(matrice_date: np.ndarray):
    """
    Ruleaza trei teste de concordanta (Shapiro-Wilk, Kolmogorov-Smirnov, Chi-patrat)
    pe fiecare coloana a matricei de date pentru a verifica normalitatea.

    - Testeaza KS impotriva distributiei normale standard N(0,1).
    - Testeaza Chi2 impotriva distributiei normale standard N(0,1).
    - Stocheaza 'True' pentru Chi2 daca p_value < 0.01 (semnificand respingerea H0).
    - Array-ul de rezultate are 4 coloane, desi doar 3 sunt folosite.

    Argumente:
        matrice_date (np.ndarray): Matricea (array NumPy 2D) de analizat.

    Returneaza:
        np.ndarray: O matrice boolean (nr_coloane, 4) cu rezultatele testelor.
                    Coloana 0: Testul Shapiro-Wilk (p > 0.1)
                    Coloana 1: Testul Kolmogorov-Smirnov (p >= 0.05)
                    Coloana 2: Testul Chi-patrat (p < 0.01)
                    Coloana 3: Neutilizata
    """
    nr_linii, nr_coloane = matrice_date.shape

    # Initializare matrice rezultate. Nota: forma este (m, 4) in codul original.
    rezultate_teste = np.empty(shape=(nr_coloane, 4), dtype=bool)

    # Itereaza prin fiecare coloana
    for idx_coloana in range(nr_coloane):
        # Extrage datele din coloana curenta
        coloana_curenta = matrice_date[:, idx_coloana]

        # --- 1. Testul Shapiro-Wilk ---
        rezultat_shapiro = shapiro(coloana_curenta)
        # Stocheaza 'True' daca p_value > 0.1
        rezultate_teste[idx_coloana, 0] = rezultat_shapiro[1] > 0.1

        # --- 2. Testul Kolmogorov-Smirnov (LOGICA ORIGINALA) ---
        # Testeaza impotriva distributiei NORMALE STANDARD ("norm" = N(0,1))
        rezultat_kstest = kstest(coloana_curenta, "norm")
        # Stocheaza 'True' daca p_value >= 0.05
        rezultate_teste[idx_coloana, 1] = rezultat_kstest[1] >= 0.05

        # --- 3. Testul Chi-patrat (LOGICA ORIGINALA) ---
        p_value_chi2 = test_chi2(coloana_curenta)
        # Stocheaza 'True' daca p_value < 0.01
        rezultate_teste[idx_coloana, 2] = p_value_chi2 < 0.01

    return rezultate_teste


def test_chi2(vector_date: np.ndarray):
    """
    Calculeaza testul de concordanta Chi-patrat (χ²).

    MENTINE LOGICA ORIGINALA:
    1. Compara cu distributia NORMALA STANDARD (N(0,1)),
       ignora media si std_dev calculate ale datelor.
    2. Foloseste 'm-1' (nr_intervale - 1) grade de libertate.
    3. Returneaza p-value calculat cu functia CDF (Cumulative Distribution Function).

    Argumente:
        vector_date (np.ndarray): Vectorul (array 1D) de date.

    Returneaza:
        float: Valoarea p (p-value) a testului, conform logicii originale.
    """
    nr_esantioane = len(vector_date)

    # Aceste variabile sunt calculate in codul original, dar NU SUNT FOLOSITE
    # in calculul frecventelor asteptate (in 'norm.cdf(limite_intervale)').
    media_vector = np.mean(vector_date)
    std_dev_vector = np.std(vector_date)

    # 1. Imparte datele in intervale (bins) folosind regula Sturges
    frecvente_observate, limite_intervale = np.histogram(vector_date, bins="sturges")

    # 'm' in codul original
    nr_intervale = len(frecvente_observate)

    # 2. Calculeaza probabilitatile asteptate PENTRU N(0,1) (LOGICA ORIGINALA)
    # Functia 'norm.cdf' fara 'loc' si 'scale' foloseste N(0,1)
    cdf_norm = norm.cdf(limite_intervale)

    probabilitati_intervale = cdf_norm[1:] - cdf_norm[:nr_intervale]

    # 3. Calculeaza frecventele asteptate (teoretice)
    frecvente_asteptate = nr_esantioane * probabilitati_intervale

    # Evita impartirea la zero
    frecvente_asteptate[frecvente_asteptate == 0] = 1

    # 4. Calculeaza statistica Chi-patrat
    statistica_chi2 = np.sum(
        (frecvente_observate - frecvente_asteptate) ** 2 / frecvente_asteptate
    )

    # 5. Calculeaza p-value (LOGICA ORIGINALA)
    # Se folosesc 'nr_intervale - 1' grade de libertate
    grade_libertate = nr_intervale - 1

    # Asiguram cel putin 1 grad de libertate
    if grade_libertate <= 0:
        grade_libertate = 1

    # Se foloseste functia CDF (Cumulative Distribution Function)
    valoare_p = chi2.cdf(statistica_chi2, grade_libertate)

    return valoare_p


def f_entropy(dataframe_date: pd.DataFrame):
    """
    Calculeaza Entropia Shannon (Indexul de diversitate) pentru fiecare coloana.
    """
    # Extrage valorile intr-o matrice NumPy (fostul 'x')
    matrice_valori = dataframe_date.values

    # Calculeaza totalul pe fiecare coloana (axis=0) (fostul 'tx')
    total_coloane = np.sum(matrice_valori, axis=0)
    # Previne impartirea la zero (logica originala)
    total_coloane[total_coloane == 0] = 1

    # Calculeaza matricea de proportii (fostul 'p')
    # p[i, j] = x[i, j] / total_coloane[j]
    proportii = matrice_valori / total_coloane

    # Gestioneaza proportiile egale cu zero (logica originala)
    proportii[proportii == 0] = 1

    # Calculeaza Entropia Shannon
    # E[j] = - Suma_i ( p[i,j] * log2(p[i,j]) )
    entropie = -np.sum(proportii * np.log2(proportii), axis=0)

    # Returneaza rezultatul ca o Serie Pandas, pastrand numele coloanelor
    return pd.Series(entropie, dataframe_date.columns)


def f_disim(dataframe_date: pd.DataFrame):
    """
    Calculeaza Indexul de Disimilaritate (ID) pentru fiecare coloana.
    """

    # Extrage valorile intr-o matrice NumPy (fostul 'x')
    matrice_valori = dataframe_date.values

    # Calculeaza totalul pe fiecare coloana (axis=0) (fostul 'tx')
    total_coloane = np.sum(matrice_valori, axis=0)
    # Previne impartirea la zero (logica originala)
    total_coloane[total_coloane == 0] = 1

    # Calculeaza totalul pe fiecare linie (axis=1) (fostul 'sx')
    total_linii = np.sum(matrice_valori, axis=1)

    # Calculeaza matricea complementara (fostul 'r')
    # r[i, j] = total_linii[i] - matrice_valori[i, j]
    matrice_complement = (total_linii - matrice_valori.T).T

    # Calculeaza totalul pe coloane al matricei complementare (fostul 'tr')
    total_coloane_complement = np.sum(matrice_complement, axis=0)
    # Previne impartirea la zero (logica originala)
    total_coloane_complement[total_coloane_complement == 0] = 1

    # Calculeaza indexul de disimilaritate (fostul 'd')
    # Formula: 0.5 * Suma( | (x / tx) - (r / tr) | ) pe coloane
    diferenta_proportii = np.abs(
        matrice_valori / total_coloane - matrice_complement / total_coloane_complement
    )
    index_disimilaritate = 0.5 * np.sum(diferenta_proportii, axis=0)

    # Returneaza rezultatul ca o Serie Pandas, pastrand numele coloanelor
    return pd.Series(index_disimilaritate, dataframe_date.columns)


def calcul_procent(serie_date: pd.Series):
    """
    Calculeaza ponderea (procentul) fiecarui element dintr-o serie
    raportat la suma totala a seriei.
    """

    # Calculeaza (valoare * 100) / suma_totala
    # Logica originala nu trateaza cazul cand suma este 0.
    return serie_date * 100 / serie_date.sum()


def acp(matrice_date: np.ndarray, ddof=0, scal=True):
    """
    Realizeaza Analiza Componentelor Principale (ACP / PCA).

    Argumente:
        matrice_date (np.ndarray): Matricea de date (observatii pe linii, variabile pe coloane).
        ddof (int): Delta Degrees of Freedom (folosit la calculul std si covarianta).
        scal (bool): Daca este True, datele sunt standardizate (se calculeaza
                     matricea de corelatie). Daca este False, datele sunt
                     doar centrate (se calculeaza matricea de covarianta).

    Returneaza:
        tuple: Contine:
            - matrice_prelucrata (np.ndarray): Datele centrate si (daca scal=True) standardizate.
            - matrice_cov_cor (np.ndarray): Matricea de covarianta (scal=False)
                                            sau corelatie (scal=True).
            - valori_proprii (np.ndarray): Valorile proprii sortate descrescator.
            - vectori_proprii (np.ndarray): Vectorii proprii sortati (componentele principale).
    """
    nr_linii, nr_coloane = matrice_date.shape

    # 1. Centrarea datelor (scaderea mediei fiecarei coloane)
    medii_coloane = np.mean(matrice_date, axis=0)
    matrice_prelucrata = matrice_date - medii_coloane

    if scal:
        # Impartirea la deviatia standard a fiecarei coloane
        std_dev_coloane = np.std(matrice_prelucrata, axis=0, ddof=ddof)
        # Se evita impartirea la zero daca o coloana are deviatie standard nula
        std_dev_coloane[std_dev_coloane == 0] = 1
        matrice_prelucrata = matrice_prelucrata / std_dev_coloane
        # matricea_prelucrata este acum standardizata(media=0, std_dev=1)

    # 3. Calculul matricei de covarianta / corelatie
    # Daca scal=True, 'matrice_prelucrata' e standardizata,
    #   deci 'matrice_cov_cor' va fi matricea de corelatie.
    # Daca scal=False, 'matrice_prelucrata' e doar centrata,
    #   deci 'matrice_cov_cor' va fi matricea de covarianta.
    matrice_cov_cor = (
            (1 / (nr_linii - ddof)) * matrice_prelucrata.T @ matrice_prelucrata
    )

    # 4. Calculul valorilor si vectorilor proprii
    valori_proprii, vectori_proprii = np.linalg.eig(matrice_cov_cor)

    # 5. Sortarea descrescatoare a valorilor si vectorilor proprii
    # 'indici_sortare' contine indicii valorilor proprii in ordine descrescatoare
    indici_sortare = np.flip(np.argsort(valori_proprii))

    # 'alpha' in codul original
    valori_proprii_sortate = valori_proprii[indici_sortare]

    # 'a' in codul original
    vectori_proprii_sortati = vectori_proprii[:, indici_sortare]

    return (
        matrice_prelucrata,
        matrice_cov_cor,
        valori_proprii_sortate,
        vectori_proprii_sortati,
    )


def tabelare_varianta(valori_proprii_sortate: np.ndarray):
    """
    Creaza un DataFrame Pandas cu sumarul variantei explicate de
    fiecare componenta principala (valoare proprie).

    Argumente:
        valori_proprii_sortate (np.ndarray): Un vector 1D cu valorile proprii
                                             sortate descrescator.

    Returneaza:
        pd.DataFrame: Un tabel cu varianta, varianta cumulata,
                      procentajul si procentajul cumulat.
    """
    # Numarul de componente (egal cu lungimea vectorului de valori proprii)
    nr_componente = len(valori_proprii_sortate)

    # Crearea indexului pentru DataFrame (ex: "C1", "C2", ...)
    etichete_componente = ["C" + str(i + 1) for i in range(nr_componente)]

    # Initializarea DataFrame-ului
    tabel_varianta = pd.DataFrame(index=etichete_componente)

    # Adaugarea coloanelor
    tabel_varianta["Varianta"] = valori_proprii_sortate
    tabel_varianta["Varianta cumulata"] = np.cumsum(valori_proprii_sortate)

    varianta_totala = sum(valori_proprii_sortate)

    tabel_varianta["Procent varianta"] = (
                                                 valori_proprii_sortate * 100
                                         ) / varianta_totala
    tabel_varianta["Procent cumulat"] = np.cumsum(tabel_varianta["Procent varianta"])

    return tabel_varianta


def salvare_ndarray(
        matrice_date: np.ndarray,
        nume_linii,
        nume_coloane,
        nume_index="",
        nume_fisier_output="out.csv",
):
    dataframe_temporar = pd.DataFrame(
        matrice_date, index=nume_linii, columns=nume_coloane
    )

    dataframe_temporar.index.name = nume_index

    dataframe_temporar.to_csv(nume_fisier_output)

    return dataframe_temporar


# Adaugă în functii.py
def calcul_calitate_si_contributii(matrice_date_std, componente, valori_proprii):
    # Calitatea reprezentării punctelor (Cos2)
    # Cos2 = Componente^2 / Distanta_la_origine^2
    dist_origine = np.sum(matrice_date_std ** 2, axis=1)
    calitate_puncte = (componente ** 2).T / dist_origine

    # Contribuția punctelor la varianța componentelor
    # Contrib = Componente^2 / (Nr_puncte * Valoare_proprie)
    n = matrice_date_std.shape[0]
    contributie_puncte = (componente ** 2) / (n * valori_proprii)

    return calitate_puncte.T, contributie_puncte


def calcul_indicatori_acp(c, alpha, r_xc):
    # 1. Cosinusuri patratice (Calitatea reprezentarii punctelor pe axe)
    # Formula: c^2 / distanta euclidiana la origine
    c2 = c * c
    cosin = (c2.T / np.sum(c2, axis=1)).T

    # 2. Contributii (Cat de mult contribuie un punct la varianta unei axe)
    # Formula: c^2 / (n * varianta_axei) sau c^2 / suma_coloana
    contrib = c2 * 100 / np.sum(c2, axis=0)

    # 3. Comunalitati (Calitatea reprezentarii variabilelor)
    # Formula: Suma patratelor corelatiilor factoriale
    r2 = r_xc * r_xc
    comm = np.cumsum(r2, axis=1)

    return cosin, contrib, comm


def tabelare_varianta_factori(varianta_tupla):
    """
    Creează un tabel sumar pentru varianta explicată de factorii comuni.
    Argument: varianta_tupla - returnată de model_af.get_factor_variance()
    """
    valori_proprii_factori = varianta_tupla[0]
    m = len(valori_proprii_factori)

    tabel_varianta = pd.DataFrame(
        data={
            "Varianta": valori_proprii_factori,
            "Varianta cumulata": np.cumsum(valori_proprii_factori),
            "Procent varianta": varianta_tupla[1] * 100,
            "Procent cumulat": varianta_tupla[2] * 100
        },
        index=["F" + str(i + 1) for i in range(m)]
    )
    return tabel_varianta


def calcul_partitie(h: np.ndarray, k=None):
    # m este numărul de fuziuni, n este numărul de obiecte inițiale
    m = h.shape[0]
    n = m + 1

    if k is None:
        # Metoda Elbow (Partiția optimală):
        # Căutăm cel mai mare salt între distanțele de fuziune succesive
        diferente = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(diferente) + 1
        k = n - j
    else:
        # Dacă k este specificat, calculăm pragul pentru acel număr de clusteri
        j = n - k

    # Calculăm threshold-ul (pragul) pentru colorarea dendrogramei
    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2

    # Inițializăm partiția ca fiind formată din obiecte individuale (singleton)
    c = np.arange(n)

    # Propagăm fuziunile în vectorul de partiție până la nivelul dorit
    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i

    # Convertim codurile numerice în etichete prietenoase (ex: C1, C2...)
    partitie = ["C" + str(i + 1) for i in pd.Categorical(c).codes]
    return k, color_threshold, np.array(partitie)