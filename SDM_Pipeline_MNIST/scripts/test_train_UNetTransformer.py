from SDM_Pipeline_MNIST.modules.UnetTransformer import *

import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers


def main(config):
    save_dir = config.app_config.save_file
    print("Save model to:{}".format(save_dir))
    logger = loggers.TensorBoardLogger(
        save_dir, name="test_UnetTR", version=1, log_graph=True
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=logger,
        check_val_every_n_epoch=50,
        max_epochs=config.model_UnetTR.UnetTR["num_epochs"],
        log_every_n_steps=10,
    )

    model = UNetTransformer(**vars(config.model_UnetTR))
    trainer.fit(model)
