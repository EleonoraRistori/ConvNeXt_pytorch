import torch
import numpy as np


def test(model, device, test_loader, criterion, val=False):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss.append(criterion(output, target).item())  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    mode = "Val" if val else "Test"
    print('\{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        mode,
        np.mean(test_loss), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = correct / len(test_loader.dataset)
    return np.mean(test_loss), test_acc


def k_correct_pred(output, target, topk=1):
    """
    Calculate the top-k accuracy for the given model output and target labels.

    Parameters:
    - output (torch.Tensor): Raw output from the model (e.g., logits or probabilities).
    - target (torch.Tensor): True labels.
    - topk (tuple): Top-k values for accuracy calculation.

    Returns:
    - list: Top-k accuracy values.
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)

        return int(correct_k.item())


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    k_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            correct += k_correct_pred(output, target, topk=1)
            k_correct += k_correct_pred(output, target, topk=5)

    print('Test set:    top-1 Accuracy: {}/{} ({:.0f}%)     top-5 Accuracy: {}/{} ({:.0f}%) \n'.format(correct,
                                                                                                       len(test_loader.dataset),
                                                                                                       100. * correct / len(
                                                                                                           test_loader.dataset),
                                                                                                       k_correct,
                                                                                                       len(test_loader.dataset),
                                                                                                       100. * k_correct / len(
                                                                                                           test_loader.dataset)))
    test_acc = correct / len(test_loader.dataset)
    return test_acc
