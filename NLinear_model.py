# CNN 모델 정의
class NLinear(nn.Module):
    def __init__(self):
        super(NLinear, self).__init__()
        # self.Linear = nn.Linear(10, 1,bias = True)
        self.Linear2 = torch.nn.Linear(16, 16,bias =True)
        self.Relu = nn.ReLU()
        self.Linear = torch.nn.ModuleList()
        for i in range(199):
            self.Linear.append(torch.nn.Linear(10, 1))
        ## a = torch.randn(32, 10, 150)
        ## torch.max(a, 1).values.size()
        

    def forward(self, x):
        
        seq_last = x[:,-1,:].unsqueeze(1) 
        #print("seq_last: {}".format(seq_last)) # seq_last: torch.Size([64, 1, 199])
        
        x = x - seq_last
        
        
        for i in range(199):
            output = torch.zeros([x.size(0), 1, 199], dtype=x.dtype)
            
            output[:,:,i] = self.Linear[i](x[:,:,i])
        x = output    
        x = torch.sum(x, axis = 2)
        x = x.squeeze()


        return x + seq_last[:,:,0].squeeze(1)

        aa = torch.mean(seq_last, axis = 2)
        aa = x + aa.squeeze(1)
        return aa

