import csv
from statistics import mean
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import numpy as np
import scipy.stats as st


# Utilier la fonte que j'aime bien
def load_font():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ["Fira Mono"]
    # Lui dire de toujours utiliser la fonte sans-serif
    matplotlib.rcParams['font.family'] = ["sans-serif"]


# Chargement des donnÃ©es depuis le CSV
def gather_data(filename):
    with open(filename, 'r') as f:
        raw_data = list(csv.reader(f, delimiter=','))
        pp_data = {}

        row_count = len(raw_data)
        for i in range(row_count):
            if raw_data[i][1] in pp_data:
                pp_data[raw_data[i][1]].append([raw_data[i][0]] + raw_data[i][2:])
            else:
                pp_data[raw_data[i][1]] = [[raw_data[i][0]] + raw_data[i][2:]]
    return pp_data


# Ordonner les donnÃ©es, calculer les intervalles de confiances Ã  95%
def process_data(pp_data):
    sizes = pp_data.keys()
    raw_accuracies = []
    sorted_data = []
    for d in pp_data.values():
        name = d[0][0]
        times = [float(dd[1]) for dd in d]
        raw_accuracies.append([float(dd[2]) for dd in d])
        sorted_data.append((name, np.mean(times), np.mean(raw_accuracies[-1])))

    # Format data
    sizes = [int(s) for s in sizes]
    accuracies = []
    names = []
    acc_conf95_lb = []
    acc_conf95_hb = []

    for r in raw_accuracies:
        conf = st.t.interval(0.95, len(r) - 1, loc=np.mean(r), scale=st.sem(r))
        acc_conf95_lb.append(conf[0])
        acc_conf95_hb.append(conf[1])

    for p in sorted_data:
        accuracies.append(p[2])
        names.append(p[0])

    couple_sorted = list(zip(*sorted(zip(sizes, accuracies, names), key=lambda x: x[0])))
    sizes = list(couple_sorted[0])
    accuracies = list(couple_sorted[1])
    names = list(couple_sorted[2])

    return sizes, names, accuracies, acc_conf95_lb, acc_conf95_hb


# Tracer la courbe bleue: les "accuracies" avec intervalles de confiance.
def plot_accu(names, accuracies, acc_conf95_lb, acc_conf95_hb):
    # Demande Ã  plt de me donner une figure et un axe.
    # Pour plt, une figure est une collection d'axes, et un axe (axes en anglais)
    # est le graph que l'on observe, composÃ© de deux axes (axis en anglais) x et y, et de la courbe
    # Utiliser cette forme plutÃ´t que le classique "plt.plot ..." me permet d'avoir plusieurs axe sur une seule figure
    # Pour simplifier, je vais utiliser les termes anglais pour axe/axis. Donc x-axis est l'axe des abscisses
    fig, ax1 = plt.subplots()
    color = 'tab:blue'

    # Changer les lÃ©gendes
    ax1.set_xlabel('Name of the layers')
    ax1.set_ylabel('Accuracy', color=color)
    # Tracer la courbe
    ax1.plot(names, accuracies, marker='o', label='accuracy/size', color=color)
    # Et les incertitudes
    ax1.fill_between(names, acc_conf95_lb, acc_conf95_hb, alpha=0.5, color=color)

    # Fixer la valeur maximale sur l'axis
    ax1.set_ylim(top=1)
    # Lui dire de ne pas afficher le fond. Un axe a aussi un fond,
    # en gÃ©nÃ©ral transparent. Dans certains (ex: export en jpg),
    # il est mis en blanc et peux faire disparaitre des courbes.
    # Pour Ã©viter les mauvaises surprises je l'efface.
    ax1.patch.set_visible(False)
    # ÃŠtre certain que les axis sont affichÃ©s. Non nÃ©cessaire, mais plus sÃ»r
    ax1.xaxis.set_visible(True)
    ax1.yaxis.set_visible(True)
    # Les graphes par dÃ©faut sont entourÃ©s d'une "boite" noire. Les quatres bords
    # de cette boite sont les "spines" (top, right, left, bottom).
    # Deux de ces bords sont utilisÃ©s pour les axes
    # J'efface les bords supÃ©rieurs et droits
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Je translate lÃ©gÃ¨rement le x-axis vers l'Ã©xterieur. Je trouve Ã§a plus esthÃ©tique
    ax1.spines['left'].set_position(('outward', 10))
    # Et je le peint en bleu
    ax1.spines['left'].set_color(color)
    # Et je peint aussi les tirets d'Ã©chelle
    ax1.tick_params(axis='y', colors=color)

    # Idem, je translate l'axe du bas vers l'Ã©xterieur. Pour la symÃ©trie.
    ax1.spines['bottom'].set_position(('outward', 10))

    # J'affiche la grille de fond
    ax1.grid(color="#d2d2d2", linestyle='dashed', linewidth=0.5)

    # Je renvoi les objets. Fig va Ãªtre nÃ©cessaire si on veux rajouter un axe
    # Et ax1 peut l'Ãªtre si on veut le lier Ã  quelque chose
    return fig, ax1


# Pas utilisÃ©.
def plot_blackbox_triple(ax1):
    # Value obtained with the script 'accuracy_and_error_bbv.py
    ax1.plot(names, [0.930016] * len(names), color='black', label='blackbox_majority_vote')
    ax1.fill_between(names, [0.9150105] * len(names), [0.945021] * len(names), color='black', alpha=0.5)


def plot_size_histogramm(fig: plt.Figure, ax1: plt.Axes, names, sizes):
    # RÃ©gler les limites d'axes
    ax1.set_ylim(bottom=0.70)
    ax1.set_xlim(left=-0.5, right=len(names) - 1 + 0.5)

    # Attention, le bazard commence. On veut faire deux choses:
    # 1. CrÃ©er un axis sur la droite, qui partage le mÃªme x-axis que notre prÃ©cÃ©dent tracÃ© (ax1)
    # 2. Et rÃ©duire la taille du y-axis pour qu'il ne monte pas jusqu'au dessus de la boite (dÃ©jÃ  partiellement effacÃ©e)

    # Pour 2, je me simplifie la tache avec un produit en croix. Je suppose que les bas sont alignÃ©s, et que je veux aligner
    # la valeur maximale de mon y-axis sur une valeur donnÃ©es du y-axis de ax1. Ici alignment=0.85. Donc le sommet de mon nouveau
    # y-axis va Ãªtre au niveau de 0.85 sur l'autre y-axis.
    #
    # Pour crÃ©er 2., on peut utiliser fig.add_axes(boite), oÃ¹ boite sont les dimensions et la position de la boite reprÃ©sentant le nouveau axe.
    # boite=(x0, y0, width, height). On veut conserver x0, y0 et width, mais pas height. D'oÃ¹ le produit en croix. (Fais un dessin sur papier
    # pour te convaincre de la formule si besoin).
    # Pour les valeurs de x0, y0, width, on ne les invente pas. On demande Ã  ax1 ses dimensions
    b, l, w, h = ax1.get_position().bounds
    # Puis les valeurs extrÃ¨mes pour faire le produit en croix.
    ymin, ymax = ax1.get_ylim()

    alignment = 0.85
    # Les bornes de mon nouveau y-axis
    (vmin, vmax) = 10 ** 0, 10 ** 6
    color = 'green'

    # On demande un nouveau axe, aux dimensions prÃ©cÃ©dement calculÃ©es.
    # Et on n'oublie pas de demander le partage avec ax1 de son x-axis.
    right_ax = fig.add_axes((b, l, w, h * (alignment - ymin) / (ymax - ymin)), sharex=ax1)
    # Et on dit Ã  matplotlib de dÃ©placer les lÃ©gendes et les tirets Ã  droite
    right_ax.yaxis.set_label_position('right')
    right_ax.yaxis.set_ticks_position('right')

    # C'est parti pour le cosmÃ©tique.
    # Comme avant, dÃ©calage vers l'Ã©xtÃ©rieur
    right_ax.spines['right'].set_position(('outward', 10))
    # Fixer les limites
    right_ax.set_ylim(vmin, vmax)
    # Passer en log-scale
    right_ax.set_yscale('log')

    # Cacher le patch, le x-axis (puisque celui de ax1 affiche dÃ©jÃ  la lÃ©gende)
    right_ax.patch.set_visible(False)
    right_ax.xaxis.set_visible(False)
    right_ax.yaxis.set_visible(True)
    # Cacher les "spines"
    right_ax.spines["top"].set_visible(False)
    right_ax.spines["bottom"].set_visible(False)
    right_ax.spines["left"].set_visible(False)

    # Colorer l'axis restant
    right_ax.spines['right'].set_color(color)
    right_ax.tick_params(axis='y', colors=color)

    # Traiter les donnÃ©es Ã  afficher
    extra_artists = []
    dif_sizes = [sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)]

    # Afficher un histogramme
    right_ax.bar(names, height=[0] + dif_sizes, alpha=0.5, color=color)

    # Et ajouter la lÃ©gende.
    # Pour chacune de ses opÃ©rations, matplotlib renvoi une rÃ©fÃ©rence vers un objet
    # gÃ©nÃ©rique appelÃ© "artist". Ce peut Ãªtre une lÃ©gende, un axe, un axis, etc.
    # D'ordinaire on s'en fiche.
    # Mais ici, j'ai besoin de le rÃ©cupÃ©rer, pour pouvoir dire Ã  matplotlib, lors du tracÃ©
    # de prendre en compte cet objet lors du calcul de la fenÃªtre minimale.
    # Par dÃ©fault, mpl rogne Ã  droite au maximum. Or cette lÃ©gende sera sur la droite.
    # On en prend donc note
    art_label = right_ax.set_ylabel("Number of weights", color=color)

    # Et j'en fais une liste au cas oÃ¹ j'aurais besoin, dans cette fonction, d'en renvoyer d'autre
    # (ce n'est pas le cas dans ce code)
    extra_artists.append(art_label)
    return extra_artists


load_font()
filename = 'norm_precision_2e6.csv'
if len(sys.argv) > 1:
    filename = sys.argv[1]

pp_data = gather_data(filename)
sizes, names, accuracies, acc_conf95_lb, acc_conf95_hb = process_data(pp_data)

# TracÃ© de la courbe bleue
fig, ax1 = plot_accu(names, accuracies, acc_conf95_lb, acc_conf95_hb)

# TracÃ© de la courbe verte et prise en note de l'artist
extra_artists = plot_size_histogramm(fig, ax1, names, sizes)

save_file = filename[:filename.rfind('.')] + '.pdf'
# Ici on lui donne les "extra artits" Ã  prendre en compte.
fig.savefig(save_file, bbox_extra_artists=extra_artists, bbox_inches='tight')

# Et voilÃ .