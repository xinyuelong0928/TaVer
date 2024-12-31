
import torch
import torch.nn as nn
import os
import glob
from datetime import datetime
import re

class BaseModel(nn.Module):
    def __init__(self, save_type=None,load_type=None):
        super(BaseModel, self).__init__()
        self.save_type = save_type
        self.load_type = load_type

        self.save_model_name = f'model_{self.save_type}' if self.save_type else 'model' 
        self.load_model_name = f'model_{self.load_type}' if self.load_type else 'model'

        self.last_epoch = 0 

    def load(self, model_dir='./checkpoints', mode=0, specified_path=None, optimizer=None):
        load_path = self._get_load_path(specified_path, self.load_model_name, model_dir, mode)
        if load_path and os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            
            model_state_dict = checkpoint.get('model_state_dict')
            if model_state_dict:
                self.load_state_dict(model_state_dict)
                print(f"Model loaded from {load_path}, epoch: {checkpoint.get('epoch')}.")
            else:
                print(f"Failed to load model_state_dict from {load_path}.")

            self.last_epoch = checkpoint.get('epoch', 0)
        else:
            print(f"No model found for {self.load_model_name} in {model_dir}, starting from scratch.")
            self.last_epoch = 0  
        return 0
    
    def transfer(self, path, strict=False):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            checkpoint_state_dict = checkpoint['model_state_dict']
            model_state_dict = self.state_dict()

            new_state_dict = {}

            missing_parameters = []
            extra_parameters = []

            for name, parameter in model_state_dict.items():
                if name in checkpoint_state_dict:
                    if checkpoint_state_dict[name].size() == parameter.size():
                        new_state_dict[name] = checkpoint_state_dict[name]
                    else:
                        extra_parameters.append(name)
                else:
                    missing_parameters.append(name)

            self.load_state_dict(new_state_dict, strict=False)
            print(f"Model parameters transferred from {path}. Successfully loaded parameters: {len(new_state_dict)}")

            if missing_parameters:
                print(f"Parameters not found in the checkpoint and using default: {missing_parameters}")
            if extra_parameters:
                print(f"Parameters in checkpoint but not used due to size mismatch: {extra_parameters}")

        else:
            print(f"No checkpoint found at {path} to transfer parameters from.")

    def save(self, epoch, optimizer=None, model_dir='./checkpoints', mode=1):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.last_epoch = epoch
        save_path = self._determine_save_path(model_dir, self.save_model_name, self.last_epoch, mode)
        
        save_dict = {'epoch': self.last_epoch, 'model_state_dict': self.state_dict()}
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(save_dict, save_path)
        print(f'Model saved to {save_path}')

    def _remove_batchnorm_state(self, checkpoint_model_state_dict):
        """Remove batch normalization layer's runtime state parameters."""
        return {k: v for k, v in checkpoint_model_state_dict.items() if "running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k}

    def _load_model_parameters(self, optimizer=None, model_dir='./checkpoints', mode=1):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.last_epoch = 0 

        save_path = self._get_load_path(specified_path=None,load_model_name=self.load_model_name, model_dir=model_dir, mode=mode)
        checkpoint = torch.load(save_path)
        checkpoint_model_state_dict = self._remove_batchnorm_state(checkpoint['model_state_dict'])
        model_state_dict = self.state_dict()
        new_model_state_dict = {}
        skipped_parameters = []

        for name, parameter in self.named_parameters():
            if name in checkpoint_model_state_dict and parameter.size() == checkpoint_model_state_dict[name].size():
                new_model_state_dict[name] = checkpoint_model_state_dict[name]
            else:
                new_model_state_dict[name] = model_state_dict[name]
                skipped_parameters.append(name)
        
        self.load_state_dict(new_model_state_dict, strict=False)
        if skipped_parameters:
            print("Skipped parameters (not loaded from checkpoint):")
            for param_name in skipped_parameters:
                print(param_name)

    def _get_load_path(self, specified_path, load_model_name, model_dir, mode):
        if specified_path:
            return specified_path
        if mode == 0:
            pattern = os.path.join(model_dir, f'{load_model_name}_epoch_*.pth')
            files = glob.glob(pattern)
            epochs = [int(re.search('epoch_([0-9]+)', f).group(1)) for f in files]
            if epochs:
                return os.path.join(model_dir, f'{load_model_name}_epoch_{max(epochs)}.pth')
        elif mode == 1:
            return os.path.join(model_dir, f'{load_model_name}.pth')
        elif mode == 2:
            files = glob.glob(os.path.join(model_dir, f'{load_model_name}_*.pth'))
            if files:
                return sorted(files, key=os.path.getmtime)[-1]
        return None

    def _determine_save_path(self, model_dir, save_model_name, last_epoch, mode):
        if mode == 0:
            return os.path.join(model_dir, f'{save_model_name}_epoch_{last_epoch}.pth')
        elif mode == 1:
            return os.path.join(model_dir, f'{save_model_name}.pth')
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            return os.path.join(model_dir, f'{save_model_name}_{timestamp}.pth')
