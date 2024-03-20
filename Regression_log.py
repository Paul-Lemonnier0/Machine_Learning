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


class Reg_log_multi(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self, d, k):
        super(Reg_log_multi, self).__init__()

        # Un seul layer lin
        self.layer = th.nn.Linear(d,k)
        self.layer.reset_parameters()

    # passe forward du modèle
    def forward(self, x):
        out = self.layer(x)
        return th.nn.functional.softmax(out,1)
    

"""

Import data

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

Modèle et déplacement GPU

"""

# On deplace nos tensors vers le cpu
device = th.device("cuda:0")

X_train = th.from_numpy(X_train).float().to(device)
Y_train = th.from_numpy(Y_train).long().to(device)

X_valid = th.from_numpy(X_valid).float().to(device)
Y_valid = th.from_numpy(Y_valid).long().to(device)


eta = 0.01
nb_epochs = 10000

d = train_data.shape[1] # shape : nb colonnes : 4096
w = np.unique(train_target).size #20 classes

model = Reg_log_multi(d, w)

criterion = th.nn.CrossEntropyLoss()

# On choisit notre optimiseur
optimizer = optim.SGD(model.parameters(), lr=eta)

model = model.to(device)



nb_epochs = 100000
pbar = tqdm(range(nb_epochs))

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

epochs = 10000
eta = 0.01

accuracy_test = 0.621
accuracy_train = 0.666
loss = 2.41

"""


"""

Entrainement modèle

"""

for i in tqdm(range(nb_epochs)):

  # remise a 0
  optimizer.zero_grad()

  # On entraine notre modele
  f_train = model(X_train)

  # On calcule la loss et son gradient
  loss = criterion(f_train,Y_train)
  loss.backward()

  # Mise a jour des poids
  optimizer.step()


"""

Test model

"""


y_pred_train = prediction(f_train)
accuracy_train = accuracy(y_pred_train.cpu(), Y_train.cpu())

f_test = model(X_valid)

y_pred_valid = prediction(f_test)
accuracy_valid = accuracy(y_pred_valid.cpu(), Y_valid.cpu())

print("accuracy_train: ", accuracy_train, "accuracy_valid:", accuracy_valid)


"""

Test hyperparametres

"""

"""
model = Reg_log_multi(d, w)

criterion = th.nn.CrossEntropyLoss()

# On envoie notre modele sur le cpu
model = model.to(device)

nb_epochs = 1000
l_eta = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

accuracies_train = []
accuracies_valid = []

# On choisit un eta
for eta in tqdm(l_eta):

  # On charge l'optimizer
  optimizer = optim.SGD(model.parameters(), lr=eta)

  # On entraine notre modele
  for i in tqdm(range(nb_epochs)):

    # remise a 0
    optimizer.zero_grad()

    # On entraine notre modele
    f_train = model(X_train)

    # On calcule la loss et son gradient
    loss = criterion(f_train,Y_train) # On pourrait par exmple afficher les loss chaque 100 itterations
    loss.backward()

    # Mise a jour des poids
    optimizer.step()

  # On calcule l'accuracy sur le train
  y_pred_train = prediction(f_train)
  accuracy_train = accuracy(y_pred_train.cpu(), Y_train.cpu())
  accuracies_train.append(accuracy_train)

  print("epoch:", nb_epochs, "eta:", eta, "train:", accuracy_train)

  """