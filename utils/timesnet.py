import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1) 
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    top_values, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, args):
        super(TimesBlock, self).__init__()
        self.seq_len = args.seq_len
        self.k = args.top_k 
        self.conv = nn.Sequential(
            Inception_Block_V1(args.d_model, args.d_ff,
                               num_kernels=args.num_kernels),
            nn.GELU(),
            Inception_Block_V1(args.d_ff, args.d_model,num_kernels=args.num_kernels)
        )
    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i] 

            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len ), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1) 
            else:
                length = self.seq_len 
                out = x
            out = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len , :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res

class Classifier(nn.Module):
    def __init__(self, config, args):
        super(Classifier, self).__init__()
        self.seq_len = args.seq_len 
        self.model = nn.ModuleList([TimesBlock(args) for _ in range(args.e_layers)])
        self.layer = args.e_layers
        self.layer_norm = nn.LayerNorm(args.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(args.dropout)
        self.projection = nn.Linear(args.d_model * args.seq_len, 1)

    def classification(self, x_enc):
        enc_out = x_enc

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1) 
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):

        dec_out = self.classification(x_enc)
        return dec_out 