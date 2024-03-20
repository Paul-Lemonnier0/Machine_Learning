import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from tqdm import tqdm
import torch.optim as optim

# fonction qui renvoie la prediction a partir de la sortie du modele (softmax(output(...)))
def prediction(sortie_modele):
    return th.argmax(sortie_modele, dim=1)

# taux d'erreur
def accuracy(pred, y):
    return np.count_nonzero(pred == y)/pred.shape[0]


class Neural_network_multi_classif(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self, d, k, h1, h2, h3):
        super(Neural_network_multi_classif, self).__init__()

        self.layer1 = th.nn.Linear(d, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        #self.layer3 = th.nn.Linear(h2, h3)
        self.layer3 = th.nn.Linear(h2, k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        #self.layer3.reset_parameters()
        self.layer4.reset_parameters()

    # passe forward du modèle
    def forward(self, x):

        #couche initiale
        l1 = th.nn.functional.sigmoid(self.layer1(x))

        #passage de la couche 1 a la couche 2
        l2 = th.nn.functional.sigmoid(self.layer2(l1))

        #l3 = th.nn.functional.sigmoid(self.layer3(l2))

        #passage à la sortie (couche 2 en entrée)
        return th.nn.functional.softmax(self.layer4(l2) ,1)

"""

IMPORT DATA

"""


chemin_fichier_train_data = "/content/kanji_train_data.csv"
chemin_fichier_train_target = "/content/kanji_train_target.csv"

# On charge nos fichiers sample de data et de target
train_target = np.genfromtxt(chemin_fichier_train_target)
train_data = np.genfromtxt(chemin_fichier_train_data, delimiter=',')

# On les sample (je n'ai pas utilisé sample car les target et features sont dans des fichiers separés)
index_sample = int(train_data.shape[0]*0.7)

X_train = train_data[:index_sample]
Y_train = train_target[:index_sample]

X_valid = train_data[index_sample:]
Y_valid = train_target[index_sample:]


"""

Model et deplacement GPU (cuda)

"""


eta = 0.01
nb_epochs = 10000
pbar = tqdm(range(nb_epochs))

device = th.device("cuda:0")

d = train_data.shape[1] # shape => nb de colonnes : 4096
w = np.unique(train_target).size #20 classes

model = Neural_network_multi_classif(d, w, 200, 100, 200)


X_train = th.from_numpy(X_train).float().to(device)
Y_train = th.from_numpy(Y_train).long().to(device)

X_valid = th.from_numpy(X_valid).float().to(device)
Y_valid = th.from_numpy(Y_valid).long().to(device)

criterion = th.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=eta)

model = model.to(device)



"""

Test du modèle

"""
for i in pbar:
    # Remise à zéro des gradients
    optimizer.zero_grad()

    f_train = model(X_train)

    loss = criterion(f_train, Y_train)
    # Calculs des gradients
    loss.backward()

    # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
    optimizer.step()

    if (i % 1000 == 0):

        y_pred_train = prediction(f_train)

        accuracy_train = accuracy(y_pred_train.cpu(), Y_train.cpu())
        loss = criterion(f_train,Y_train)

        f_test = model(X_valid)
        y_pred_test = prediction(f_test)

        accuracy_test = accuracy(y_pred_test.cpu(), Y_valid.cpu())

        pbar.set_postfix(iter=i, loss = loss.item(), accuracy_train=accuracy_train, accuracy_test=accuracy_test)

"""

RESULTAT :

2 layers :

epochs = 100000
eta = 0.01

accuracy_test = 0.765
accuracy_train = 0.666
loss = 2.26


3 layers :

epochs = 10000
eta = 0.01

accuracy_test = 0.696
accuracy_train = 0.734
loss = 2.34


"""