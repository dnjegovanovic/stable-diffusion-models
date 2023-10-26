from SDM_Pipeline_MNIST.modules.SimpleUnetModules import *

import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers

def main():
    save_dir=r"./simple_unet_test"
    logger = loggers.TensorBoardLogger(save_dir, name="logs", version=1, log_graph=True)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[0],
                         logger=logger,
                         check_val_every_n_epoch=50,
                         max_epochs=50,
                         log_every_n_steps=10)
    
    model = SimpleUNetModules()
    trainer.fit(model)
