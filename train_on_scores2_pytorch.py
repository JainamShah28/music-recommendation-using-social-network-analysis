
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.scaler = StandardScaler()
        self.inputs = self.scaler.fit_transform(df[['Bias_U', 'Bias_V', 'Bias_U_div_Bias_V', 'Jaccard_Similarity', 'Normalised_Weight']].values)
        self.labels = df['Actual_Val'].values
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, label_tensor

class MyModule(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import gc
    import pandas as pd
    
    # Load the data into a pandas dataframe
    df = pd.read_csv('scores2.csv', error_bad_lines=False)

    # Remove any unnecessary columns
    df = df.dropna()
    df = df.rename(columns={'Bias_U': 'Bias_U', 'Bias_V': 'Bias_V', 'Bias_U/Bias_V': 'Bias_U_div_Bias_V', 'Jaccard Similarity': 'Jaccard_Similarity', 'Normalised_Weight': 'Normalised_Weight', 'Actual_Val': 'Actual_Val'})

    # Create a MyDataset object
    dataset = MyDataset(df)
    del df
    gc.collect()

    # Split the dataset into train and validation sets
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for train and validation sets
    batch_size = 2000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MyModule(5)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_loader, val_loader)

    # Save the PyTorch Lightning model
    torch.save(model.state_dict(), 'train_on_scores2.pt')