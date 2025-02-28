import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mobilenet import MobileNetV2ForCIFAR8M
from DataLoad import DataLoad
from train import Train
from test import Test
import sys
import os
from Log import Logger

class BaseModle:
    def __init__(self, config, model_name):
        self._config = config
        self._model_name=model_name
                # set Directories
        self._train_path = self._config.directories["train_path"]  
        self._test_path = self._config.directories["test_path"]
        self._save_path= self._config.directories["save_path"]
        self._save_graph=self._config.directories["output_graph"]
        self._save_log=self._config.directories["save_log"]
        
        self._log=Logger(self._save_log,"model1")
        
        # set hyperparameters
        self._batch_size = self._config.hyperparameters["batch_size"]
        self._learning_rate = self._config.hyperparameters["learning_rate"]
        self._epoch = self._config.hyperparameters["epochs"]
        self._valdata_ratio=self._config.hyperparameters["valdata_ratio"]
        self._width_transform=self._config.hyperparameters["width_transform"]
        self._drop_out=self._config.hyperparameters["drop_out"]
        self._height_transform=self._config.hyperparameters["height_transform"]

        # set parameters
        self._num_classes=self._config.model_parameters["num_classes"]
        self._device=self._config.model_parameters["device"]

    def save_result(self,model,tr_ac,val_ac,tr_los,val_los):
        
        #save trained model weight
        save_dir = os.path.dirname(self._save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), self._save_path)
        
        #save validation and train graph
        fig,ax=plt.subplots(ncols=2)
        ax[0].plot(tr_ac, label="Train_ac")
        ax[0].plot(val_ac, label="val_ac")
        ax[0].legend()
        ax[0].set_title(f"{self._model_name} Accuracy")
        ax[1].set_title(f"{self._model_name} loss values")
        ax[1].plot(tr_los, label="Train_los")
        ax[1].plot(val_los, label="val_los")
        ax[1].legend()
        
        # Save the figure
        plt.savefig(self._save_graph, dpi=300, bbox_inches='tight')  # High-quality save
    
    def main(self):

        self._log.log("======== Starting Model Training ========")

        # Load dataset
        self._log.log("Loading dataset...")
        Loader = DataLoad(
            self._train_path, self._test_path, self._valdata_ratio, 
            self._batch_size, self._height_transform, self._width_transform
        )
        train_loader, val_loader, test_loader = Loader.DataLoad()
        self._log.log("Dataset loaded successfully!")

        # Initialize model
        self._log.log("Initializing the model...")
        model=MobileNetV2ForCIFAR8M(self._num_classes,self._height_transform,self._width_transform,self._drop_out)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log.log(f"Model initialized with {trainable_params:,} trainable parameters.")

        # Log model architecture
        self._log.log(f"Model Architecture: \n{model}")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)

        # Train model
        self._log.log(f"Starting training for {self._epoch} epochs...")
        train = Train(model, self._epoch, train_loader, val_loader, criterion, optimizer, self._device,self._log)
        tr_ac, val_ac, tr_los, val_los = train.train_model()

        # Save results
        self._log.log("Saving trained model and training results...")
        self.save_result(model, tr_ac, val_ac, tr_los, val_los)

        # Test model
        self._log.log("Starting model evaluation...")
        test = Test(model, test_loader, criterion, self._device,self._log)
        test.test_model()

        self._log.log("======== Model Training Completed! ========")    