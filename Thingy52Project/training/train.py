import torch
from CNN import CNN
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import CSVDataModule

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


def main():
    path = "data"
    k = 3
    window_size = 3 # seconds
    sampling_frequency = 60 # Hz
    data_module = CSVDataModule(
        root_dir=path,
        batch_size=32,
        k_folds=k,
        window_size=window_size, # seconds
    )

    data_module.setup()

    # Perform k-fold cross-validation
    for fold in range(data_module.k):
        print(f'Fold {fold + 1}/{data_module.k + 1}')

        model = CNN(window_size * sampling_frequency, fold + 1, classes_names=data_module.legend)

        # Define checkpoint callback to save the best model for each fold
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/fold_{fold + 1}',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=20,
            accelerator="gpu", # cpu
            devices=1,
            logger=True,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback]
        )

        # Train the model
        trainer.fit(model, data_module.train_dataloader(fold=fold), data_module.val_dataloader(fold=fold))

        # Test the model
        trainer.test(model, data_module.val_dataloader(fold=fold))


if __name__ == '__main__':
    main()