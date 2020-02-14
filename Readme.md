## TP Reinforcement Learning

Les TP 1, 9 et 11 sont dans des les dossiers du même nom, car un peu exceptionnels par rapport aux autres qui suivent tous la même structure. Le dossier `tp-rl` contient lui tous les codes pour les différents agents de RL.

## Structure des dossiers
```
.
├── data
├── report
├── tp1
├── tp11
│   ├── results
│   └── src
│       ├── architectures
│       └── utils
├── tp9
└── tp-rl			# Tous les scripts arena_*.py correspondent aux tests des agents
    └── agents			# Contient le code des agents
	├── a2c.py
	├── ddpg2.py
	├── ddpg.py
	├── deepqlearning.py
	├── dynaq.py
	├── iterPolicy.py
	├── iterValue.py
	├── maddpg.py
	├── ppo.py
	└── qlearning.py	
```

## Différents TP

Pour les TP de tp-rl, le dossier signifiant est le dossier `agents/` qui contient les implémentations des différents agents, tous suivant une API similaire. Les fichiers `tp-rl/arena_*.py` correspondent aux évaluations des différents agents. Concernant MADDP j'en fournis une implémentation en pytorch ici, ainsi qu'une implémentation en Tensorflow réalisée l'an dernier ici https://github.com/icannos/collaborativecells

Le TP 9 contient uniquement le notebook fourni lors du TP, les résultats sont présentés dans le rapport.
Le TP 11 contient non seulement l'implémentation d'un VAE usuel mais aussi du smooth VAE issu du papier: https://openreview.net/forum?id=H1gfFaEYDS . C'était notre projet d'AMAL avec Pierre Marion et Ariane Marandon.

