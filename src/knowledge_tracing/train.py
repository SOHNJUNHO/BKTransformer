import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAUROC


class NeuralBKTLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr

        # Metrics
        #self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()
    
    def forward(self, obs, output):
        return self.model(obs, output)
    
    def training_step(self, batch, batch_idx):
        obs, output, keys = batch
        corrects, latents, params, loss = self(obs, output)
        batch_size = obs.size(0)

        with torch.no_grad():  # Don't need gradients for metrics
            corrects = corrects.squeeze(-1)
            mask = output[..., 1] != -1000
            valid_preds = corrects[mask]
            valid_targets = output[..., 1][mask]
        
            train_acc = ((valid_preds > 0.5).float() == valid_targets).float().mean()
 
        self.log(
            'train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            'train_acc',
            train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        obs, output, keys = batch
        corrects, latents, params, loss = self(obs, output)
        batch_size = obs.size(0)
        
        mask = output[..., 1] != -1000
        corrects = corrects.squeeze(-1) 
    
         # Get valid predictions and targets
        valid_preds = corrects[mask]
        valid_targets = output[..., 1][mask]
        
        val_acc = ((valid_preds > 0.5).float() == valid_targets).float().mean()
        self.val_auc.update(valid_preds, valid_targets.long())
        
        self.log(
            'val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            'val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            'val_auc',
            self.val_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
    
    
    def test_step(self, batch, batch_idx):
        obs, output, keys = batch
        corrects, latents, params, loss = self(obs, output)
        batch_size = obs.size(0)
        
        mask = output[..., 1] != -1000
        corrects = corrects.squeeze(-1)
        
        valid_preds = corrects[mask]
        valid_targets = output[..., 1][mask]
        
        test_acc = ((valid_preds > 0.5).float() == valid_targets).float().mean()
        self.test_auc.update(valid_preds, valid_targets.long())
        
        # Log test metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
