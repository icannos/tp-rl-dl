import torch

def train(epoch, model, train_loader, log_interval=10, device=torch.device("cpu"), accuracy=False):
    # model.train() ?
    train_loss = 0
    n_batch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        n_batch += 1
        data = data.to(device).view(data.shape[0], data.shape[2]*data.shape[3])
        loss = model.training_step(data, target.to(device))
        train_loss += loss
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tObjective: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss))
            if accuracy:
                prediction = model(data)
                acc = torch.sum(prediction.argmax(dim=1) == target).item() / target.shape[0]
                print('Accuracy: {:.3f}'.format(acc))
        
    print('====> Epoch: {} Average objective: {:.4f}'.format(
          epoch, train_loss / n_batch))