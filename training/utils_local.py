from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from sklearn.metrics import average_precision_score

def get_country_to_region():
    return {
        'Angola': 0, 
        'Botswana': 0, 
        'Cameroon': 0,
        'Egypt': 0, 
        'Ghana': 0, 
        'Nigeria': 0, 
        'South_Africa': 0,
        'Argentina': 1, 
        'Brazil': 1, 
        'Colombia': 1, 
        'Mexico': 1, 
        'Uruguay': 1, 
        'Venezuela': 1, 
        'China': 2, 
        'Hong_Kong': 2, 
        'Japan': 2,
        'South_Korea':2, 
        'Taiwan': 2, 
        'Bulgaria': 3, 
        'France': 3, 
        'Greece': 3, 
        'Italy': 3, 
        'Poland': 3, 
        'Romania': 3, 
        'Switzerland': 3, 
        'United_Kingdom': 3,
        'Cyprus': 3, 
        'Germany': 3, 
        'Ireland': 3, 
        'Netherlands': 3, 
        'Portugal': 3, 
        'Spain': 3, 
        'Ukraine': 3, 
        'Indonesia': 4, 
        'Malaysia': 4, 
        'Philippines': 4, 
        'Singapore': 4, 
        'Thailand': 4, 
        'Jordan': 5, 
        'Saudi_Arabia': 5, 
        'Turkey': 5, 
        'United_Arab_Emirates': 5, 
        'Yemen': 5}

def get_reg_to_number():
    return {
        'Africa':0,
        'Americas': 1, 
        'EastAsia': 2, 
        'Europe': 3, 
        'SouthEastAsia': 4, 
        'WestAsia': 5}



def train_linear_with_reg(Xtrain, ytrain, Xval, yval, regs = [10, 1, 0.1, 0.01, 0.001]):
    
    best_clf = None
    best_acc = 0

    for c in regs:
        clf = LogisticRegression(C = c)
        clf.fit(Xtrain, ytrain)

        acc = clf.score(Xval, yval)

        if acc>best_acc:    
            best_acc = acc
            best_clf = clf

    print(best_acc)
    return best_clf, best_acc


def train_pytorch_linear(Xtrain, ytrain, Xval, yval, curr_attr):
    
    model = torch.nn.Linear(2048, 40)

    Xtrain = torch.Tensor(Xtrain)
    ytrain = torch.Tensor(ytrain).to(dtype=torch.long)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    batch = 512
    N = len(Xtrain)//batch + 1

    model.train()
    model.cuda()
    best_model = None
    best_acc = 0

    for e in range(200):
        model.train()
        model.cuda()
        
        for t in range(N):
            Xbatch = Xtrain[t*batch:(t+1)*batch].to(torch.device('cuda'))
            ybatch = ytrain[t*batch:(t+1)*batch].to(torch.device('cuda'))

            
            scores = model(Xbatch)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer.zero_grad()
            loss = criterion(scores, ybatch)
            loss.backward()

            optimizer.step()

        model.eval()
        scores = model(torch.Tensor(Xval).to(torch.device('cuda'))).detach().cpu().numpy()
        
        acc = average_precision_score(np.where(np.array(yval)==curr_attr, 1, 0).squeeze(), scores[:, curr_attr].squeeze())

        if acc>best_acc:
            best_acc = acc
            best_model = model.cpu().state_dict()
    
    return best_model, best_acc

