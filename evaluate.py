import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(test_dataloader, model, history, device, name):
    train_acc,test_acc,train_loss,test_loss = history[0], history[1], history[2], history[3]
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_target = []
    all_pred = []

    for data, target in test_dataloader:
        print('========================== Start Evaluating ==========================')
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target).sum().item()
        total_correct += correct
        total_samples += data.size(0)
        all_target.extend(target.cpu().numpy())
        all_pred.extend(predicted.cpu().numpy())


    patterns = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
                "Loc", "Random", "Scratch", "Near-full"]
    f1_scores = f1_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average=None)
    precision_scores = precision_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average=None)
    recall_scores = recall_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average=None)

    # ==================================== Print scores for each class ====================================
    for i, (f1, prec, rec) in enumerate(zip(f1_scores, precision_scores, recall_scores)):
        print(f'{patterns[i]:20}:F1 Score: {100 * f1:.3f}%, Precision: {100 * prec:.3f}%, Recall: {100 * rec:.3f}%')
    df = {'f1_scores':f1_scores, 'precision':precision_scores, 'recall':recall_scores}
    print("=======================================================")

    # ==================================== Calculate and print overall scores ====================================
    f1_macro = np.mean(f1_scores)
    f1_micro = f1_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average='micro')

    print(f'Overall f1_macro: {100 * f1_macro:.3f}%, f1_micro: {100 * f1_micro:.3f}%')

    # ==================================== Calculate and print Accuracy ====================================
    f1_scores = list(f1_scores)
    precision_scores = list(precision_scores)
    recall_scores = list(recall_scores)

    accuracy = 100 * total_correct / total_samples
    print(f'Accuracy: {accuracy:.3f}%')
    f1_scores.extend([f1_macro, f1_micro, accuracy])
    precision_scores.extend(['' for _ in range(3)])
    recall_scores.extend(['' for _ in range(3)])
    df = {'f1_scores':f1_scores, 'precision':precision_scores, 'recall':recall_scores}
    patterns.extend(["f1 macro", "f1 micro", "accuracy"])
    df = pd.DataFrame(df, index=patterns)
    df.to_csv("Exp_result.csv")

    # ==================================== Plot Confusion Matrix ====================================
    patterns = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
                "Loc", "Random", "Scratch", "Near-full"]
    conf_matrix = confusion_matrix(all_target, all_pred)
    accuracy_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=patterns, yticklabels=patterns)
    plt.title("Confusion Matrix")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()