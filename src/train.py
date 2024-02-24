from dataset import *
from sklearn.model_selection import train_test_split

dataset = wineDataset(
    "data/wine.data",
    transform=toTensor(),
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

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, random_state=42)
dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=True, random_state=42)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, random_state=42)
