import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.autograd as autograd
from .Base import BaseModel
import logging
import math
from .timesnet import *


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x

    
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)
        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = x.permute(0, 2, 1)  
        conved = self.convs(embedded)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))   
        return self.fc(cat)


class TimesnetClassificationSeq(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.d_size = self.args.d_size
        self.dense = nn.Linear(2 * self.d_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.W_w = nn.Parameter(torch.Tensor(2 * config.hidden_size, 2 * config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(2 * config.hidden_size, 1))
        self.linear = nn.Linear(self.args.filter_size * config.hidden_size, self.d_size)
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, 3, bidirectional=True, batch_first=True,
                           dropout=config.hidden_dropout_prob)

        # CNN
        self.window_size = self.args.cnn_size
        self.filter_size = []
        for i in range(self.args.filter_size):
            i = i+1
            self.filter_size.append(i)

        self.cnn = TextCNN(config.hidden_size, self.window_size, self.filter_size, self.d_size, 0.2)
        self.linear_mlp = nn.Linear(6*config.hidden_size, self.d_size)
        self.linear_multi = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = Classifier(config, args)

    def forward(self, seq_embeds, **kwargs):
        batch_size = seq_embeds.shape[0] // 3

        seq_embeds = seq_embeds.view(batch_size, 3, 400, 768)  
        outputs = []
        for t in range(400):  
            path_features = seq_embeds[:, :, t, :] 
            cnn_output = self.cnn(path_features) 
            global_features = self.linear(path_features.reshape(batch_size, -1))
            x_t_fused = torch.cat((cnn_output, global_features), dim=-1) 
            x_t_fused = self.dropout(x_t_fused)
            x_t_fused = self.dense(x_t_fused)
            outputs.append(x_t_fused)
        outputs = torch.stack(outputs, dim=1)  
        x= self.classifier(outputs)

        return x



class Adapter(nn.Module):
    def __init__(self, adapter_dim, embed_dim):
        super(Adapter, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_project = nn.Linear(embed_dim, adapter_dim, bias=False)
        self.up_project = nn.Linear(adapter_dim, embed_dim, bias=False)

    def forward(self, z):
        normalized_z = self.layer_norm(z)
        h = F.relu(self.down_project(normalized_z))
        return self.up_project(h) + z


low_resource_languages = ["java","php","python"]
low_resource_adapter_dim = 64
high_resource_adapter_dim = 128


class Model(BaseModel):
    def __init__(self, encoder, config, tokenizer, languages, args):
        super(Model, self).__init__(save_type=args.save_type,load_type = args.load_type) 
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.linear = nn.Linear(3, 1) 


        self.timesnetclassifier = TimesnetClassificationSeq(config, self.args)



        if args.fusion_languages:
            args.fusion_languages = args.fusion_languages.split("_")
        self.fusion_languages = sorted(args.fusion_languages) if args.fusion_languages else sorted(languages)

        self.use_adapters = args.use_adapters
        if self.use_adapters:
            self.adapters = torch.nn.ModuleDict()
            for lang in languages:
                if lang in low_resource_languages:
                    adapter_dim = low_resource_adapter_dim
                else:
                    adapter_dim = high_resource_adapter_dim           
                self.adapters[lang] = Adapter(adapter_dim, config.hidden_size)            
        else:
            self.adapters = None

        if self.use_adapters:
            for p in self.parameters():
                p.requires_grad = False
            self.enable_adapter_training(args.target_language)


    def enable_adapter_training(self, specified_languages=None):
        # Unfreeze the adapter parameters
        enable_languages = specified_languages

        logging.warning(f"Unfreezing the adapter parameters of {enable_languages}")
        for p in self.adapters[enable_languages].parameters():
            p.requires_grad = True

    def forward(self, seq_ids=None, input_ids=None, labels=None, language = None):
        batch_size = seq_ids.shape[0]
        seq_len = seq_ids.shape[1]
        token_len = seq_ids.shape[-1]

        seq_inputs = seq_ids.reshape(-1, token_len) 
        seq_embeds = self.encoder(seq_inputs, attention_mask=seq_inputs.ne(1))[0] 
        

        if self.use_adapters:
            seq_embeds = self.adapters[language](seq_embeds)  

        logits_path = self.timesnetclassifier(seq_embeds)

        prob_path = torch.sigmoid(logits_path)
        prob = prob_path
        if labels is not None:
            labels = labels.float()
            main_loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            main_loss = -main_loss.mean()

            return main_loss, prob
        else:
            return prob
