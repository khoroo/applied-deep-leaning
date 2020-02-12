from torch import nn

def fetch_model(mode):

    if mode == 'MLMC':
        linear_layer_shape = 26048
    else:
        linear_layer_shape = 15488

    model = nn.Sequential(
        #Layer 1
        nn.Conv2d(1,32,(3,3), stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        #Layer 2
        nn.Conv2d(32,32,(3,3),stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(p=0.5),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1)),
        #Layer 3
        nn.Conv2d(32,64,(3,3),stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        #Layer 4
        nn.Conv2d(64,64,(3,3),stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(p=0.5),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,1)),
        nn.Flatten(),
        #Layer 5
        nn.Linear(linear_layer_shape, 1024),
        nn.Sigmoid(),
        nn.Dropout(p=0.5),
        #Layer 6
        nn.Linear(1024,10)
    )
    
    return model
                        