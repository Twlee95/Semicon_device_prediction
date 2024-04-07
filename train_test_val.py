import torch
from metric import pred_acc, R2_acc, SL_acc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(model, trainloader, NLinear_optimizer, loss_fn):
    model.train()
    model.zero_grad()
    train_loss = 0.0
    for i, (X, y) in enumerate(trainloader):

  
        X = X.float().to(device)

        y = y.float().to(device)
        
        # print(X.size()) # torch.Size([64, 10, 199])
        # print(y.size()) # torch.Size([64, 1, 1])
        
        y_pred = model(X)
        
        #decoder_hidden = encoder_hidden
        y_true = y[:, :].float().to(device)  

        loss = loss_fn(y_pred.view(-1), y_true.view(-1))  

        loss.backward()  ## gradient 계산
        NLinear_optimizer.step()  

        train_loss += loss.item()  ## item()은 loss의 스칼라값을 칭하기때문에 cpu로 다시 넘겨줄 필요가 없다.
        
        #print("Epoch: {}, Batch: {}, Train Loss: {}".format(ep+1, i+1, loss))

    train_loss = train_loss / len(trainloader)
    
    return model, train_loss


def validation(model, valloader, NLinear_optimizer, loss_fn):
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):

    
            X = X.float().to(device)

            y = y.float().to(device)

            y_pred = model(X)
            #decoder_hidden = encoder_hidden
            y_true = y[:, :].float().to(device)  

            loss = loss_fn(y_pred.view(-1), y_true.view(-1))  
            
            val_loss += loss.item()  ## item()은 loss의 스칼라값을 칭하기때문에 cpu로 다시 넘겨줄 필요가 없다.
            
            #print("Epoch: {}, Batch: {}, Train Loss: {}".format(ep+1, i+1, loss))

        val_loss = val_loss / len(valloader)
        
    #rint("#######################################################")
    return model, val_loss

def test(model, testloader):
    model.eval()
    
    Predacc_metric = 0.0
    r2_metric = 0.0
    sl_metric = 0.0
    
    graph_pred = np.array([])
    graph_true = np.array([])
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X = X.float().to(device)
            # encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]
            y = y.float().to(device)

            y_pred= model(X)

            y_true = y[:, :].float().to(device)
            
            
            acc = pred_acc(y_pred.view(-1), y_true.view(-1))
            r2 = R2_acc(y_true.view(-1),y_pred.view(-1))
            sl = SL_acc(y_pred.view(-1), y_true.view(-1))

            
            
            graph_pred = np.concatenate((graph_pred,y_pred.view(-1)))
            graph_true = np.concatenate((graph_true,y_true.view(-1)))
            
            Predacc_metric += acc
            r2_metric += r2
            sl_metric += sl
            
            print("pred_acc: {}".format(acc))
            print("pred_r2: {}".format(r2))
            print("pred_sl: {}".format(sl))
            
        Predacc_metric = Predacc_metric / len(testloader)
        r2_metric = r2_metric / len(testloader)
        sl_metric = sl_metric / len(testloader)
        

    print("mean pred_acc: {}".format(Predacc_metric))
    return Predacc_metric,r2_metric,sl_metric, graph_pred, graph_true