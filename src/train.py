from dataset import *
from model import *
from sklearn.model_selection import train_test_split

dataset = wineDataset(
    "data/wine.data",
    transform=OneHotEncoding(3),
    headers=[
        "class",
        "alcohol",
        "malicacid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "0D280_0D315_of_diluted_wines",
        "proline",
    ],
)

train_data, test_data = train_test_split(
    dataset, test_size=0.2, shuffle=True, random_state=42
)
test_data, dev_data = train_test_split(
    test_data, test_size=0.5, shuffle=True, random_state=42
)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

hidden_dims = 32
num_epochs = 200
lr = 0.0001

model = MultilayerPerceptronClassifier(13, hidden_dims, 3)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fct = nn.CrossEntropyLoss()

model.learn(train_dataloader, dev_dataloader, num_epochs, optimizer, loss_fct)
model.evaluate(test_dataloader, loss_fct, "./predictions.csv")
