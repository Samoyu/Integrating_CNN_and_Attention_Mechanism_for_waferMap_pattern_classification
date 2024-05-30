import torch
import torch.nn as nn
import time
from tqdm import tqdm

def training(train_dataloader, model, epochs, learning_rate, device, optimizer_choice):
    num_epochs = epochs
    period = 5
    criterion = nn.CrossEntropyLoss()

    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_choice == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    start_time = time.time()
    for epoch in range(num_epochs):
        print('========================== Start Training ==========================')
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data, label in tqdm(train_dataloader):
            data, target = data.to(device), label.to(device)
            
            if optimizer_choice in ['adam', 'sgd']:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            elif optimizer_choice in ['lbfgs']:
                def closure():
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                outputs = model(data)
                loss = criterion(outputs, target)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == target).sum().item()
            total_correct += correct
            total_samples += data.size(0)

        train_loss.append(total_loss / len(train_dataloader))
        train_acc.append(100 * total_correct / total_samples)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}%')

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"execution time: {execution_time} s")

    return model, (train_acc, test_acc, train_loss, test_loss)