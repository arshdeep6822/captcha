import torch
from torch import nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3,3), padding=(1,1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)
        
        self.gru = nn.GRU(64, 32, num_layers=2, bidirectional=True, dropout = 0.25)
        self.output = nn.Linear(64, num_chars+1)
    
    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        
        x = F.relu(self.conv_1(images))
        x = self.max_pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.max_pool_2(x)
        
        x = x.permute(0, 3, 1, 2)  # Rearranging dimensions for RNN input
        x = x.view(bs, x.size(1), -1)
        x = self.linear_1(x)
        x = self.drop_1(x)
        
        x, _ = self.gru(x)
        
        x = self.output(x)
        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, num_classes)
        
        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)
        
            input_lengths = torch.full(
                size=(bs,),
                fill_value=x.size(0),  # seq_len from the output (x.size(0))
                dtype=torch.int32
            )
            
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.int32)
            
            loss = nn.CTCLoss(blank=0)(
                log_softmax_values,
                targets,
                input_lengths,
                target_lengths
            )
            return x, loss


        
        
    
    
if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand(5, 3, 75, 300)
    targets = torch.randint(1, 20, (5,5))
    x, loss = cm(img, targets)
        