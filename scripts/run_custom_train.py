import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import click


@click.command()
@click.option('--ckpt', default='ckpt/', help='Directory for storing data')
def main(ckpt):

    import sys
    import os
    sys.path.append(os.path.dirname('..'))

    from src.data import data_loader
    from src.train import train_model
    from src.model import ResNet18


    num_classes = 10
    num_epochs = 25
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, flush=True)


    # Finetuned model
    print("Finetuned", flush=True)
    model_conv = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    # Freeze training for all layers exept the last one for finetuning
    for param in model_conv.parameters(): param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)


    model_conv = model_conv.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9) # Set small learning rate for finetuning
    exp_lr_scheduler = None

    datasets = data_loader(data_dir='data/', batch_size=batch_size)
    dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'test']}


    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, device, datasets, dataset_sizes, num_epochs=num_epochs)
    torch.save(model_conv.state_dict(), f'{ckpt}baseline_resnet_best.pth')



    # Custom model
    print("Custom", flush=True)
    model = ResNet18(num_classes).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model.fc.parameters(), lr=0.1, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)


    datasets = data_loader(data_dir='data/', batch_size=batch_size)
    dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'test']}



    best_custom_resnet = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, device, datasets, dataset_sizes, num_epochs=num_epochs)
    torch.save(best_custom_resnet.state_dict(), f'{ckpt}custom_resnet_best.pth')


    print("Done", flush=True)


if __name__ == '__main__':
    main()