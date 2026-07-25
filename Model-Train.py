import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

for epoch in range(25):

    optimizer.zero_grad()

    outputs = model(train_tensor)

    loss = criterion(outputs, target_tensor)

    loss.backward()

    optimizer.step()

    print(epoch, loss.item())