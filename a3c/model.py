from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init
import subprocess

class A3Cmicropolis6x6conv(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=20, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis6x6conv, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        self.conv_0 = nn.Conv2d(14, 32, 7, stride=1, padding=3) # 6*6*16 = 576
        self.conv_1 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
       #self.conv_2 = nn.Conv2d(16, 16, 5, stride=1, padding=2)
       #self.conv_3 = nn.Conv2d(16, 16, 7, stride=1, padding=3)
       #self.conv_4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
       #self.conv_5 = nn.Conv2d(16, 16, 5, stride=1, padding=2)

        self.actor_conv = nn.Conv2d(32, 8, 3, stride=1, padding=1)
        self.critic_linear = nn.Linear(1152, 1)


        
        #######################################################################

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_0.weight.data.mul_(relu_gain)
        self.conv_1.weight.data.mul_(relu_gain)
       
       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)


        self.train()


    def getMemorySizes(self):
        return []


    def forward(self, inputs):

            inputs, (hx, cx) = inputs
            x = inputs
            x = F.relu(self.conv_0(x))
            x = F.relu(self.conv_1(x))
           #x = F.elu(self.conv_2(x))
           #x = F.elu(self.conv_3(x))
           #x = F.elu(self.conv_4(x))
           #x = F.elu(self.conv_5(x))
        
            action = self.actor_conv(x)
            action = action.view(1, -1)
            x = x.view(1, -1)
            value = self.critic_linear(x) 
            return value, action, ([], [])


class A3Cmicropolis10x10conv(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=10, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis10x10conv, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        self.conv_0 = nn.Conv2d(14, 32, 7, stride=1, padding=3) # 6*6*16 = 576
        self.conv_1 = nn.Conv2d(32, 64, 7, stride=1, padding=3)
        self.conv_2 = nn.Conv2d(64, 32, 7, stride=1, padding=3)
        self.conv_3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
       #self.conv_2 = nn.Conv2d(16, 16, 5, stride=1, padding=2)
       #self.conv_3 = nn.Conv2d(16, 16, 7, stride=1, padding=3)
       #self.conv_4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
       #self.conv_5 = nn.Conv2d(16, 16, 5, stride=1, padding=2)

        self.actor_conv = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.critic_conv = nn.Conv2d(16, 8, 3, 1, 1)


        
        #######################################################################

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_0.weight.data.mul_(relu_gain)
        self.conv_1.weight.data.mul_(relu_gain)
        self.conv_2.weight.data.mul_(relu_gain)
        self.conv_3.weight.data.mul_(relu_gain)
       
       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
       #self.critic_linear.weight.data = norm_col_init(
       #    self.critic_linear.weight.data, 1.0)
       #self.critic_linear.bias.data.fill_(0)


        self.train()


    def getMemorySizes(self):
        return []


    def forward(self, inputs):

            inputs, (hx, cx) = inputs
            x = inputs
            x = F.relu(self.conv_0(x))
            x = F.relu(self.conv_1(x))
            x = F.relu(self.conv_2(x))
            x = F.relu(self.conv_3(x))
           #x = F.elu(self.conv_2(x))
           #x = F.elu(self.conv_3(x))
           #x = F.elu(self.conv_4(x))
           #x = F.elu(self.conv_5(x))
        
            action = self.actor_conv(x)
            action = action.view(1, -1)
            value = self.critic_conv(x)
            value = value.view(1, -1)
            return value, action, ([], [])



class A3Cmicropolis20x20conv(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=10, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis20x20conv, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        self.conv_0 = nn.Conv2d(14, 16, 3, 1, 1)
        self.conv_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_7 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_9 = nn.Conv2d(16, 16, 3, 1, 1)    
        self.conv_10 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_11 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_12 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_13 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_14 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_15 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_16 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_17 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_18 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_19 = nn.Conv2d(16, 16, 3, 1, 1)
        
        self.actor_conv = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.critic_conv = nn.Conv2d(16, 8, 3, 1, 1)


        
        #######################################################################

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')


       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
       #self.critic_linear.weight.data = norm_col_init(
       #    self.critic_linear.weight.data, 1.0)
       #self.critic_linear.bias.data.fill_(0)


        self.train()


    def getMemorySizes(self):
        return []


    def forward(self, inputs):

            inputs, (hx, cx) = inputs
            x = inputs
            x = F.relu(self.conv_0(x))
            x = F.relu(self.conv_1(x))
            x = F.relu(self.conv_2(x))
            x = F.relu(self.conv_3(x))
            x = F.relu(self.conv_4(x))
            x = F.relu(self.conv_5(x))
            x = F.relu(self.conv_6(x))
            x = F.relu(self.conv_7(x))
            x = F.relu(self.conv_8(x))
            x = F.relu(self.conv_9(x))
            x = F.relu(self.conv_10(x))
            x = F.relu(self.conv_11(x))
            x = F.relu(self.conv_12(x))
            x = F.relu(self.conv_13(x))
            x = F.relu(self.conv_14(x))
            x = F.relu(self.conv_15(x))
            x = F.relu(self.conv_16(x))
            x = F.relu(self.conv_17(x))
            x = F.relu(self.conv_18(x))
            x = F.relu(self.conv_19(x))

        
            action = self.actor_conv(x)
            action = action.view(1, -1)
            value = self.critic_conv(x)
            value = value.view(1, -1)
            return value, action, ([], [])

class A3Cmicropolis20x20convAlinC(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=10, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis20x20convAlinC, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        self.conv_0 = nn.Conv2d(14, 16, 3, 1, 1)
        self.conv_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_7 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_8 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_9 = nn.Conv2d(16, 16, 3, 1, 1)    
        self.conv_10 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_11 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_12 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_13 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_14 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_15 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_16 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_17 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_18 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_19 = nn.Conv2d(16, 16, 3, 1, 1)

        self.critic_conv_0 = nn.Conv2d(16, 32, 10, 5, 0)
        
        self.actor_conv = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.critic_linear = nn.Linear(288, 1)


        
        #######################################################################

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')


       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
       #self.critic_linear.weight.data = norm_col_init(
       #    self.critic_linear.weight.data, 1.0)
       #self.critic_linear.bias.data.fill_(0)


        self.train()


    def getMemorySizes(self):
        return []


    def forward(self, inputs):

            inputs, (hx, cx) = inputs
            x = inputs
            x = F.relu(self.conv_0(x))
            x = F.relu(self.conv_1(x))
            x = F.relu(self.conv_2(x))
            x = F.relu(self.conv_3(x))
            x = F.relu(self.conv_4(x))
            x = F.relu(self.conv_5(x))
            x = F.relu(self.conv_6(x))
            x = F.relu(self.conv_7(x))
            x = F.relu(self.conv_8(x))
            x = F.relu(self.conv_9(x))
            x = F.relu(self.conv_10(x))
            x = F.relu(self.conv_11(x))
            x = F.relu(self.conv_12(x))
            x = F.relu(self.conv_13(x))
            x = F.relu(self.conv_14(x))
            x = F.relu(self.conv_15(x))
            x = F.relu(self.conv_16(x))
            x = F.relu(self.conv_17(x))
            x = F.relu(self.conv_18(x))
            x = F.relu(self.conv_19(x))

        
            action = self.actor_conv(x)
            action = action.view(1, -1)
            x = self.critic_conv_0(x)
            x = x.view(1, -1)
            value = self.critic_linear(x)
            return value, action, ([], [])


#class A3Cmicropolis(torch.nn.Module):
#
#    def __init__(self, num_inputs, action_space, 
#             map_width=20, num_obs_channels=14, num_tools=8):
#
#        super(A3Cmicropolis, self).__init__()
#
#        from ConvLSTMCell import ConvLSTMCell
#
#        num_lstm_channels = num_obs_channels + 2 # 16
#        num_outputs = num_tools * map_width * map_width
#
#        self.conv_in = nn.Conv2d(14, 16, 
#                                 1, stride=1, padding=0) # 20*20*16 = 6400
#        self.conv_lstm1 = ConvLSTMCell(16, 16)
#        self.conv_lstm2 = ConvLSTMCell(16, 16)
#        self.conv_lstm3 = ConvLSTMCell(16, 16)
#        self.linear0 = nn.Linear(6400, 1024)
#        self.lin_lstm = nn.LSTMCell(1024, 1024)
#        self.actor_linear = nn.Linear(1024, 3200)
#        self.critic_linear = nn.Linear(1024, 1)
#
#        self.conv_lstm1.weight = self.conv_lstm1.Gates.weight
#        self.conv_lstm1.bias = self.conv_lstm1.Gates.bias
#        self.conv_lstm1._grad = self.conv_lstm1.Gates.weight._grad        
#        self.conv_lstm2.weight = self.conv_lstm2.Gates.weight
#        self.conv_lstm2.bias = self.conv_lstm2.Gates.bias
#        self.conv_lstm2._grad = self.conv_lstm2.Gates.weight._grad
#        self.conv_lstm3.weight = self.conv_lstm3.Gates.weight
#        self.conv_lstm3.bias = self.conv_lstm3.Gates.bias
#        self.conv_lstm3._grad = self.conv_lstm3.Gates.weight._grad
#        
#        #######################################################################
#
#        self.apply(weights_init)
#        relu_gain = nn.init.calculate_gain('relu')
#        self.conv_in.weight.data.mul_(relu_gain)
#       
#        self.actor_linear.weight.data = norm_col_init(
#            self.actor_linear.weight.data, 0.01)
#        self.actor_linear.bias.data.fill_(0)
#        self.critic_linear.weight.data = norm_col_init(
#            self.critic_linear.weight.data, 1.0)
#        self.critic_linear.bias.data.fill_(0)
#
#        self.lin_lstm.bias_ih.data.fill_(0)
#        self.lin_lstm.bias_hh.data.fill_(0)
#
#        self.train()
#
#
#    def getMemorySizes(self):
#        return [(1, 16, 20, 20) 
#                for i in range(3)] + [(1, 1024)]
#
#
#    def forward(self, inputs):
#
#            inputs, (hx, cx) = inputs
#            x = inputs
#            x = F.elu(self.conv_in(x))
#            hx0, cx0 = self.conv_lstm1(x, (hx[0], cx[0]))
#            hx1, cx1 = self.conv_lstm2(hx0, (hx[1], cx[1]))
#            hx2, cx2 = self.conv_lstm3(hx1, (hx[2], cx[2]))
#            x = hx2
#            x = torch.tanh(self.linear0(x.view(x.size(0), -1)))
#            hx3, cx3 = self.lin_lstm(x, (hx[3], cx[3]))
#            x = hx3
#            value = self.critic_linear(x) 
#            action = self.actor_linear(x)
#            return value, action, ([hx0, hx1, hx2, hx3], [cx0, cx1, cx2, cx3])

     

class A3CmicropolisSqueeze(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=20, num_obs_channels=14, num_tools=8):

        super(A3CmicropolisSqueeze, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        num_lstm_channels = num_obs_channels + 2 # 16
        num_outputs = num_tools * map_width * map_width

        self.conv_in = nn.Conv2d(num_obs_channels, 16, 
                                 1, stride=1, padding=0) # 20*20*16 = 6400
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv2 = nn.Conv2d(16,  64,
                              5, stride=2, padding=0) # 8*8*64 = 4096
        self.conv3 = nn.Conv2d(64, 128,
                               3, stride=2, padding=0) # 3*3*128 = 1152 
        self.linear = nn.Linear(1152, 512)
        self.critic_linear = nn.Linear(512, 1) 
        self.lin_lstm = nn.LSTMCell(512, 512)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 
                                          2, stride=1, padding=0) # 3*3*128 = 1152
        self.deconv2 = nn.ConvTranspose2d(128, 32,
                                          3, stride=2, padding=0) # 7*7*32 = 1568
        self.deconv3 = nn.ConvTranspose2d(32, 8,
                                          8, stride=2, padding=0) # 20*20*8 = 3200
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_integrate = nn.Conv2d(22, 16,
                7, stride=1, padding=3)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_preaction_0 = nn.Conv2d(16, 16,11, stride=1, padding = 5)
        self.conv_preaction_1 = nn.Conv2d(16, 16,21, stride=1, padding = 10)
       #self.linear_out_0 = nn.Linear(3200, 1600)
       #self.linear_out_1 = nn.Linear(1600, 1024)
       #self.linear_out_2 = nn.Linear(1024, 512)
       #self.linear_out_3 = nn.Linear(512, 1024)
       #self.actor_linear = nn.Linear(1024, 3200)
        self.actor_conv = nn.Conv2d(16, 8, 39, stride = 1, padding=19)

        self.conv_lstm1.weight = self.conv_lstm1.Gates.weight
        self.conv_lstm1.bias = self.conv_lstm1.Gates.bias
        self.conv_lstm1._grad = self.conv_lstm1.Gates.weight._grad        
        self.conv_lstm2.weight = self.conv_lstm2.Gates.weight
        self.conv_lstm2.bias = self.conv_lstm2.Gates.bias
        self.conv_lstm2._grad = self.conv_lstm2.Gates.weight._grad
        self.conv_lstm3.weight = self.conv_lstm3.Gates.weight
        self.conv_lstm3.bias = self.conv_lstm3.Gates.bias
        self.conv_lstm3._grad = self.conv_lstm3.Gates.weight._grad
        self.conv_lstm4.weight = self.conv_lstm3.Gates.weight
        self.conv_lstm4.bias = self.conv_lstm3.Gates.bias
        self.conv_lstm4._grad = self.conv_lstm3.Gates.weight._grad
        

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_in.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.deconv1.weight.data.mul_(relu_gain)
        self.deconv2.weight.data.mul_(relu_gain)
        self.deconv3.weight.data.mul_(relu_gain)
        self.conv_integrate.weight.data.mul_(relu_gain)
        self.conv_preaction_0.weight.data.mul_(relu_gain)
        self.conv_preaction_1.weight.data.mul_(relu_gain)
       #self.actor_conv.weight.data.mul_(relu_gain)

       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 20, 20) 
                for i in range(2)] + [(1, 512)] +  [(1, 16, 20, 20) for i in range(2)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[0], cx[0]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[1], cx[1]))
            x = hx1
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = torch.tanh(self.linear(x.view(x.size(0), -1)))
            hx2, cx2 = self.lin_lstm(x, (hx[2], cx[2]))
            x = hx2                
            value = self.critic_linear(x) 
            x = x.view(x.size(0), 128, 2, 2)
            x = F.relu(self.deconv1(x))
            x = F.relu(self.deconv2(x))
            x = F.relu(self.deconv3(x))
            x = torch.cat((x, inputs), 1)
            x = F.relu(self.conv_integrate(x))

            x = F.relu(self.conv_preaction_0(x))
            x = F.relu(self.conv_preaction_1(x))
            hx3, cx3 = self.conv_lstm3(x, (hx[3], cx[3]))
            x = hx3
            hx4, cx4 = self.conv_lstm4(x, (hx[4], cx[4]))
            x = hx4
            action = self.actor_conv(x)
            action = action.view(action.size(0), -1)
           #x = x.view(x.size(0), -1)
           #x = torch.tanh(self.linear_out_0(x))
           #x = torch.tanh(self.linear_out_1(x))
           #x = torch.tanh(self.linear_out_2(x))
           #x = torch.tanh(self.linear_out_3(x))
           #action = self.actor_linear(x)
            return value, action, ([hx0, 
                 hx1, hx2, 
                 hx3, hx4
                ], [cx0, 
                     cx1, cx2, 
                     cx3, cx4
                    ])

class LinearMicropolis(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=20, num_obs_channels=14, num_tools=8):

        super(LinearMicropolis, self).__init__()

        self.linear0 = nn.Linear(5600, 3200)
        self.critic_linear = nn.Linear(3200, 1)    
        self.actor_linear = nn.Linear(3200, 3200)
        ###########################################################

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return []


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs.view(inputs.size(0), -1)
            x = torch.tanh(self.linear0(x))           
            value = self.critic_linear(x)
            action = self.actor_linear(x)
            return value, action, ([], [])


class A3Cmicropolis(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=10, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        num_lstm_channels = num_obs_channels + 2 # 16
        num_outputs = num_tools * map_width * map_width

        self.conv_in = nn.Conv2d(num_obs_channels, 16, 
                                 1, stride=1, padding=0)                  
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_lstm5 = ConvLSTMCell(16, 16)

        self.linear_out_0 = nn.Linear(6400, 1024)
        self.linear_out_1 = nn.Linear(1024, 1024)
       #self.critic_conv = nn.Conv2d(16, 1, map_width, padding=0)
        self.critic_linear = nn.Linear(1024, 1)
    #self.actor_conv = nn.Conv2d(16, 8, 21, stride=1, padding=10)
        self.actor_linear = nn.Linear(1024, 3200)

        ###########################################################

        self.conv_lstm1.weight = self.conv_lstm1.Gates.weight
        self.conv_lstm1.bias = self.conv_lstm1.Gates.bias
        self.conv_lstm1._grad = self.conv_lstm1.Gates.weight._grad        
        self.conv_lstm2.weight = self.conv_lstm2.Gates.weight
        self.conv_lstm2.bias = self.conv_lstm2.Gates.bias
        self.conv_lstm2._grad = self.conv_lstm2.Gates.weight._grad
        self.conv_lstm3.weight = self.conv_lstm3.Gates.weight
        self.conv_lstm3.bias = self.conv_lstm3.Gates.bias
        self.conv_lstm3._grad = self.conv_lstm3.Gates.weight._grad
        self.conv_lstm4.weight = self.conv_lstm4.Gates.weight
        self.conv_lstm4.bias = self.conv_lstm4.Gates.bias
        self.conv_lstm4._grad = self.conv_lstm4.Gates.weight._grad
        self.conv_lstm5.weight = self.conv_lstm5.Gates.weight
        self.conv_lstm5.bias = self.conv_lstm5.Gates.bias
        self.conv_lstm5._grad = self.conv_lstm5.Gates.weight._grad

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_in.weight.data.mul_(relu_gain)
       #self.critic_conv.weight.data.mul_(relu_gain)
       #self.actor_conv.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 20, 20) 
                for i in range(5)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[0], cx[0]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[1], cx[1]))
            hx2, cx2 = self.conv_lstm3(hx1, (hx[2], cx[2]))
            hx3, cx3 = self.conv_lstm4(hx2, (hx[3], cx[3]))
            hx4, cx4 = self.conv_lstm5(hx3, (hx[4], cx[4]))
            x = hx4.view(x.size(0), -1)
           #hx0_f = hx0.view(hx0.size(0), -1)
           #hx1_f = hx1.view(hx1.size(0), -1)
           #x = torch.cat((hx0_f, hx1_f), 1)
            x = torch.tanh(self.linear_out_0(x))
            x = torch.tanh(self.linear_out_1(x))
            value = self.critic_linear(x)
            action = self.actor_linear(x)
           #x = hx4
           #value = self.critic_conv(x)
           #value = value.view(value.size(0), -1)
           #action = self.actor_conv(x)
           #action = action.view(action.size(0), -1)
            return value, action, ([hx0,
                                 hx1, 
                                 hx2, hx3, hx4
                                ], 
                                   [cx0, 
                                       cx1, 
                                       cx2, cx3, cx4
                                       ])

class A3Cmicropolis10x10convAC(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=10, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis10x10convAC, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        num_lstm_channels = num_obs_channels + 2 # 16
        num_outputs = num_tools * map_width * map_width

        self.conv_in = nn.Conv2d(num_obs_channels, 16, 1, stride=1, padding=0)                  
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_lstm5 = ConvLSTMCell(16, 16)
       #self.conv_lstm6 = ConvLSTMCell(16, 16)
       #self.conv_lstm7 = ConvLSTMCell(16, 16)
       #self.conv_lstm8 = ConvLSTMCell(16, 16)
       #self.conv_lstm9 = ConvLSTMCell(16, 16)
       #self.conv_lstm10 = ConvLSTMCell(16, 16)

       #self.conv_out = nn.Conv2d(16, 8, 1, 1, padding=0)                  
        self.actor_conv = ConvLSTMCell(8, 8, 1, 0)
       #self.critic_conv = ConvLSTMCell(8, 8, 11, padding=5)

        self.linear_out_0 = nn.Linear(1600, 800)
       #self.linear_out_1 = nn.Linear(1024, 800)
       #self.critic_conv = nn.Conv2d(16, 1, map_width, padding=0)
        self.critic_linear = nn.Linear(800, 1)
    #self.actor_conv = nn.Conv2d(16, 8, 21, stride=1, padding=10)
       #self.actor_linear = nn.Linear(1600, 800)

        ###########################################################

        self.conv_lstm1.weight = self.conv_lstm1.Gates.weight
        self.conv_lstm1.bias = self.conv_lstm1.Gates.bias
        self.conv_lstm1._grad = self.conv_lstm1.Gates.weight._grad        
        self.conv_lstm2.weight = self.conv_lstm2.Gates.weight
        self.conv_lstm2.bias = self.conv_lstm2.Gates.bias
        self.conv_lstm2._grad = self.conv_lstm2.Gates.weight._grad
        self.conv_lstm3.weight = self.conv_lstm3.Gates.weight
        self.conv_lstm3.bias = self.conv_lstm3.Gates.bias
        self.conv_lstm3._grad = self.conv_lstm3.Gates.weight._grad
        self.conv_lstm4.weight = self.conv_lstm4.Gates.weight
        self.conv_lstm4.bias = self.conv_lstm4.Gates.bias
        self.conv_lstm4._grad = self.conv_lstm4.Gates.weight._grad
        self.conv_lstm5.weight = self.conv_lstm5.Gates.weight
        self.conv_lstm5.bias = self.conv_lstm5.Gates.bias
        self.conv_lstm5._grad = self.conv_lstm5.Gates.weight._grad
       #self.conv_lstm6.weight = self.conv_lstm1.Gates.weight
       #self.conv_lstm6.bias = self.conv_lstm1.Gates.bias
       #self.conv_lstm6._grad = self.conv_lstm1.Gates.weight._grad        
       #self.conv_lstm7.weight = self.conv_lstm2.Gates.weight
       #self.conv_lstm7.bias = self.conv_lstm2.Gates.bias
       #self.conv_lstm7._grad = self.conv_lstm2.Gates.weight._grad
       #self.conv_lstm8.weight = self.conv_lstm3.Gates.weight
       #self.conv_lstm8.bias = self.conv_lstm3.Gates.bias
       #self.conv_lstm8._grad = self.conv_lstm3.Gates.weight._grad
       #self.conv_lstm9.weight = self.conv_lstm4.Gates.weight
       #self.conv_lstm9.bias = self.conv_lstm4.Gates.bias
       #self.conv_lstm9._grad = self.conv_lstm4.Gates.weight._grad
       #self.conv_lstm10.weight = self.conv_lstm5.Gates.weight
       #self.conv_lstm10.bias = self.conv_lstm5.Gates.bias
       #self.conv_lstm10._grad = self.conv_lstm5.Gates.weight._grad
        self.actor_conv.weight = self.actor_conv.Gates.weight
        self.actor_conv.bias = self.actor_conv.Gates.bias
        self.actor_conv._grad = self.actor_conv.Gates.weight._grad
       #self.critic_conv.weight = self.actor_conv.Gates.weight
       #self.critic_conv.bias = self.actor_conv.Gates.bias
       #self.critic_conv._grad = self.actor_conv.Gates.weight._grad


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_in.weight.data.mul_(relu_gain)
        self.conv_lstm1.weight.data.mul_(relu_gain)
        self.conv_lstm2.weight.data.mul_(relu_gain)
        self.conv_lstm3.weight.data.mul_(relu_gain)
        self.conv_lstm4.weight.data.mul_(relu_gain)
       #self.conv_lstm5.weight.data.mul_(relu_gain)
       #self.conv_lstm6.weight.data.mul_(relu_gain)
       #self.conv_lstm7.weight.data.mul_(relu_gain)
       #self.conv_lstm8.weight.data.mul_(relu_gain)
       #self.conv_lstm9.weight.data.mul_(relu_gain)
       #self.conv_lstm10.weight.data.mul_(relu_gain)
       #self.conv_out.weight.data.mul_(relu_gain)
       #self.actor_conv.weight.data.mul_(relu_gain)
       #self.critic_conv.weight.data.mul_(relu_gain)


       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 10, 10) 
                for i in range(5)] + [(1, 8, 10, 10) for i in range(1)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[0], cx[0]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[1], cx[1]))
            hx2, cx2 = self.conv_lstm3(hx1, (hx[2], cx[2]))
            hx3, cx3 = self.conv_lstm4(hx2, (hx[3], cx[3]))
            hx4, cx4 = self.conv_lstm5(hx3, (hx[4], cx[4]))
           #hx5, cx5 = self.conv_lstm6(hx4, (hx[5], cx[5]))
           #hx6, cx6 = self.conv_lstm7(hx5, (hx[6], cx[6]))
           #hx7, cx7 = self.conv_lstm8(hx6, (hx[7], cx[7]))
           #hx8, cx8 = self.conv_lstm9(hx7, (hx[8], cx[8]))
           #hx9, cx9 = self.conv_lstm10(hx8, (hx[0], cx[0]))
            x = hx4
            x = x.view(1, -1)
            x = torch.tanh(self.linear_out_0(x))
            x = x.view(1, 8, 10, 10)
           #x = torch.tanh(self.linear_out_1(x))
           #action = self.actor_linear(x)
           #x = hx4
           #x = torch.tanh(self.conv_out(x))
            hx10, cx10 = self.actor_conv(x, (hx[-1], cx[-1]))
            action = hx10
            action = action.view(action.size(0), -1)
           #hx11, cx11 = self.critic_conv(x, (hx[-1], cx[-1]))
           #value = hx11
            value = hx10
            value = value.view(value.size(0), -1)
            value = self.critic_linear(value)
            return value, action, ([
                                hx0,
                                 hx1, 
                                 hx2, hx3, hx4,
                                #hx5, hx6, hx7, hx8, hx9,
                                 hx10#hx11

                                ], 
                                   [
                                       cx0, 
                                       cx1, 
                                       cx2, cx3, cx4,
                                      #cx5, cx6, cx7, cx8, cx9,
                                       cx10#cx11
                                       ])


class A3Cmicropolis10x10linAC(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=10, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis10x10linAC, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        num_lstm_channels = num_obs_channels + 2 # 16
        num_outputs = num_tools * map_width * map_width

        self.conv_in = nn.Conv2d(14, 16, 1, stride=1, padding=0)                  
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_lstm5 = ConvLSTMCell(16, 16)
      # self.conv_lstm6 = ConvLSTMCell(16, 16)
      # self.conv_lstm7 = ConvLSTMCell(16, 16)
      # self.conv_lstm8 = ConvLSTMCell(16, 16)
      # self.conv_lstm9 = ConvLSTMCell(16, 16)
      # self.conv_lstm10 = ConvLSTMCell(16, 16)

        self.conv_out = nn.Conv2d(16, 8, 3, stride=1, padding=1)                  
        self.linear_out_0 = nn.Linear(800, 512)
       #self.deconv1 = nn.ConvTranspose2d(128, 128, 
       #                                  2, stride=1, padding=0) # 3*3*128 = 1152
       #self.deconv2 = nn.ConvTranspose2d(128, 32,
       #                                  3, stride=2, padding=0) # 7*7*32 = 1568
       #self.deconv3 = nn.ConvTranspose2d(32, 8,
       #                                  8, stride=2, padding=0) # 20*20*8 = 3200

        self.linear_out_1 = nn.Linear(512, 800)

        self.critic_linear = nn.Linear(800, 1)
        self.actor_linear = nn.Linear(800, 800)

       #self.critic_conv = nn.Conv2d(16, 1, map_width, padding=0)
      # self.critic_linear = nn.Linear(800, 1)
    #self.actor_conv = nn.Conv2d(16, 8, 21, stride=1, padding=10)
       #self.actor_linear = nn.Linear(1600, 800)

        ###########################################################

        self.conv_lstm1.weight = self.conv_lstm1.Gates.weight
        self.conv_lstm1.bias = self.conv_lstm1.Gates.bias
        self.conv_lstm1._grad = self.conv_lstm1.Gates.weight._grad        
        self.conv_lstm2.weight = self.conv_lstm2.Gates.weight
        self.conv_lstm2.bias = self.conv_lstm2.Gates.bias
        self.conv_lstm2._grad = self.conv_lstm2.Gates.weight._grad
        self.conv_lstm3.weight = self.conv_lstm3.Gates.weight
        self.conv_lstm3.bias = self.conv_lstm3.Gates.bias
        self.conv_lstm3._grad = self.conv_lstm3.Gates.weight._grad
        self.conv_lstm4.weight = self.conv_lstm4.Gates.weight
        self.conv_lstm4.bias = self.conv_lstm4.Gates.bias
        self.conv_lstm4._grad = self.conv_lstm4.Gates.weight._grad
        self.conv_lstm5.weight = self.conv_lstm5.Gates.weight
        self.conv_lstm5.bias = self.conv_lstm5.Gates.bias
        self.conv_lstm5._grad = self.conv_lstm5.Gates.weight._grad
       #self.conv_lstm6.weight = self.conv_lstm1.Gates.weight
       #self.conv_lstm6.bias = self.conv_lstm1.Gates.bias
       #self.conv_lstm6._grad = self.conv_lstm1.Gates.weight._grad        
       #self.conv_lstm7.weight = self.conv_lstm2.Gates.weight
       #self.conv_lstm7.bias = self.conv_lstm2.Gates.bias
       #self.conv_lstm7._grad = self.conv_lstm2.Gates.weight._grad
       #self.conv_lstm8.weight = self.conv_lstm3.Gates.weight
       #self.conv_lstm8.bias = self.conv_lstm3.Gates.bias
       #self.conv_lstm8._grad = self.conv_lstm3.Gates.weight._grad
       #self.conv_lstm9.weight = self.conv_lstm4.Gates.weight
       #self.conv_lstm9.bias = self.conv_lstm4.Gates.bias
       #self.conv_lstm9._grad = self.conv_lstm4.Gates.weight._grad
       #self.conv_lstm10.weight = self.conv_lstm5.Gates.weight
       #self.conv_lstm10.bias = self.conv_lstm5.Gates.bias
       #self.conv_lstm10._grad = self.conv_lstm5.Gates.weight._grad
       #self.actor_conv.weight = self.actor_conv.Gates.weight
       #self.actor_conv.bias = self.actor_conv.Gates.bias
       #self.actor_conv._grad = self.actor_conv.Gates.weight._grad
       #self.critic_conv.weight = self.actor_conv.Gates.weight
       #self.critic_conv.bias = self.actor_conv.Gates.bias
       #self.critic_conv._grad = self.actor_conv.Gates.weight._grad


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_in.weight.data.mul_(relu_gain)
        self.conv_lstm1.weight.data.mul_(relu_gain)
        self.conv_lstm2.weight.data.mul_(relu_gain)
        self.conv_lstm3.weight.data.mul_(relu_gain)
        self.conv_lstm4.weight.data.mul_(relu_gain)
        self.conv_lstm5.weight.data.mul_(relu_gain)
       #self.conv_lstm6.weight.data.mul_(relu_gain)
       #self.conv_lstm7.weight.data.mul_(relu_gain)
       #self.conv_lstm8.weight.data.mul_(relu_gain)
       #self.conv_lstm9.weight.data.mul_(relu_gain)
       #self.conv_lstm10.weight.data.mul_(relu_gain)
       #self.conv_out.weight.data.mul_(relu_gain)
       #self.actor_conv.weight.data.mul_(relu_gain)
       #self.critic_conv.weight.data.mul_(relu_gain)


       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
       #self.critic_linear.weight.data = norm_col_init(
       #    self.critic_linear.weight.data, 1.0)
       #self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 10, 10) 
                for i in range(5)] # + [(1, 8, 10, 10) for i in range(2)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[2], cx[2]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[1], cx[1]))
            hx2, cx2 = self.conv_lstm3(hx1, (hx[0], cx[0]))
            hx3, cx3 = self.conv_lstm4(hx2, (hx[3], cx[3]))
            hx4, cx4 = self.conv_lstm5(hx3, (hx[4], cx[4]))
           #hx5, cx5 = self.conv_lstm6(hx4, (hx[5], cx[5]))
           #hx6, cx6 = self.conv_lstm7(hx5, (hx[6], cx[6]))
           #hx7, cx7 = self.conv_lstm8(hx6, (hx[7], cx[7]))
           #hx8, cx8 = self.conv_lstm9(hx7, (hx[8], cx[8]))
           #hx9, cx9 = self.conv_lstm10(hx8, (hx[9], cx[9]))
            x = hx4
            x = torch.tanh(self.conv_out(x))
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.linear_out_0(x))
            x = torch.tanh(self.linear_out_1(x))
           #action = self.actor_linear(x)
           #x = hx4
            value = self.critic_linear(x)
            action = self.actor_linear(x)
           #hx10, cx10 = self.actor_conv(x, (hx[10], cx[10]))
           #action = hx10
           #action = action.view(action.size(0), -1)
           #hx11, cx11 = self.critic_conv(x, (hx[11], cx[11]))
           #value = hx11
           #value = value.view(value.size(0), -1)
            return value, action, ([
                                 hx0,
                                 hx1, 
                                 hx2, hx3, hx4,
                                #hx5, hx6, hx7, hx8, hx9,
                                #hx10, hx11

                                ], 
                                [ 
                                 cx0, 
                                 cx1, 
                                 cx2, cx3, cx4,
                                #cx5, cx6, cx7, cx8, cx9,
                                #cx10, cx11
                                ])

class A3Cmicropolis14x14linAC(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=14, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis14x14linAC, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        num_lstm_channels = num_obs_channels + 2 # 16
        num_outputs = num_tools * map_width * map_width

        self.conv_in = nn.Conv2d(14, 16, 1, stride=1, padding=0)                  
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_lstm5 = ConvLSTMCell(16, 16)
      # self.conv_lstm6 = ConvLSTMCell(16, 16)
      # self.conv_lstm7 = ConvLSTMCell(16, 16)
      # self.conv_lstm8 = ConvLSTMCell(16, 16)
      # self.conv_lstm9 = ConvLSTMCell(16, 16)
      # self.conv_lstm10 = ConvLSTMCell(16, 16)

        self.conv_out = nn.Conv2d(16, 8, 3, stride=1, padding=1)                        
        self.conv_1 = nn.Conv2d(8, 8, 3, 1, 0)
        self.linear_out_0 = nn.Linear(1152, 512)
       #self.deconv1 = nn.ConvTranspose2d(128, 128, 
       #                                  2, stride=1, padding=0) # 3*3*128 = 1152
       #self.deconv2 = nn.ConvTranspose2d(128, 32,
       #                                  3, stride=2, padding=0) # 7*7*32 = 1568
       #self.deconv3 = nn.ConvTranspose2d(32, 8,
       #                                  8, stride=2, padding=0) # 20*20*8 = 3200

        self.linear_out_1 = nn.Linear(512, 800)


        self.critic_linear = nn.Linear(800, 1)
        self.actor_linear = nn.Linear(800, 1568)

       #self.critic_conv = nn.Conv2d(16, 1, map_width, padding=0)
      # self.critic_linear = nn.Linear(800, 1)
    #self.actor_conv = nn.Conv2d(16, 8, 21, stride=1, padding=10)
       #self.actor_linear = nn.Linear(1600, 800)

        ###########################################################

        self.conv_lstm1.weight = self.conv_lstm1.Gates.weight
        self.conv_lstm1.bias = self.conv_lstm1.Gates.bias
        self.conv_lstm1._grad = self.conv_lstm1.Gates.weight._grad        
        self.conv_lstm2.weight = self.conv_lstm2.Gates.weight
        self.conv_lstm2.bias = self.conv_lstm2.Gates.bias
        self.conv_lstm2._grad = self.conv_lstm2.Gates.weight._grad
        self.conv_lstm3.weight = self.conv_lstm3.Gates.weight
        self.conv_lstm3.bias = self.conv_lstm3.Gates.bias
        self.conv_lstm3._grad = self.conv_lstm3.Gates.weight._grad
        self.conv_lstm4.weight = self.conv_lstm4.Gates.weight
        self.conv_lstm4.bias = self.conv_lstm4.Gates.bias
        self.conv_lstm4._grad = self.conv_lstm4.Gates.weight._grad
        self.conv_lstm5.weight = self.conv_lstm5.Gates.weight
        self.conv_lstm5.bias = self.conv_lstm5.Gates.bias
        self.conv_lstm5._grad = self.conv_lstm5.Gates.weight._grad
       #self.conv_lstm6.weight = self.conv_lstm1.Gates.weight
       #self.conv_lstm6.bias = self.conv_lstm1.Gates.bias
       #self.conv_lstm6._grad = self.conv_lstm1.Gates.weight._grad        
       #self.conv_lstm7.weight = self.conv_lstm2.Gates.weight
       #self.conv_lstm7.bias = self.conv_lstm2.Gates.bias
       #self.conv_lstm7._grad = self.conv_lstm2.Gates.weight._grad
       #self.conv_lstm8.weight = self.conv_lstm3.Gates.weight
       #self.conv_lstm8.bias = self.conv_lstm3.Gates.bias
       #self.conv_lstm8._grad = self.conv_lstm3.Gates.weight._grad
       #self.conv_lstm9.weight = self.conv_lstm4.Gates.weight
       #self.conv_lstm9.bias = self.conv_lstm4.Gates.bias
       #self.conv_lstm9._grad = self.conv_lstm4.Gates.weight._grad
       #self.conv_lstm10.weight = self.conv_lstm5.Gates.weight
       #self.conv_lstm10.bias = self.conv_lstm5.Gates.bias
       #self.conv_lstm10._grad = self.conv_lstm5.Gates.weight._grad
       #self.actor_conv.weight = self.actor_conv.Gates.weight
       #self.actor_conv.bias = self.actor_conv.Gates.bias
       #self.actor_conv._grad = self.actor_conv.Gates.weight._grad
       #self.critic_conv.weight = self.actor_conv.Gates.weight
       #self.critic_conv.bias = self.actor_conv.Gates.bias
       #self.critic_conv._grad = self.actor_conv.Gates.weight._grad


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_in.weight.data.mul_(relu_gain)
        self.conv_lstm1.weight.data.mul_(relu_gain)
        self.conv_lstm2.weight.data.mul_(relu_gain)
        self.conv_lstm3.weight.data.mul_(relu_gain)
        self.conv_lstm4.weight.data.mul_(relu_gain)
        self.conv_lstm5.weight.data.mul_(relu_gain)
       #self.conv_lstm6.weight.data.mul_(relu_gain)
       #self.conv_lstm7.weight.data.mul_(relu_gain)
       #self.conv_lstm8.weight.data.mul_(relu_gain)
       #self.conv_lstm9.weight.data.mul_(relu_gain)
       #self.conv_lstm10.weight.data.mul_(relu_gain)
       #self.conv_out.weight.data.mul_(relu_gain)
       #self.actor_conv.weight.data.mul_(relu_gain)
       #self.critic_conv.weight.data.mul_(relu_gain)


       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
       #self.critic_linear.weight.data = norm_col_init(
       #    self.critic_linear.weight.data, 1.0)
       #self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 14, 14) 
                for i in range(5)] # + [(1, 8, 10, 10) for i in range(2)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[2], cx[2]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[1], cx[1]))
            hx2, cx2 = self.conv_lstm3(hx1, (hx[0], cx[0]))
            hx3, cx3 = self.conv_lstm4(hx2, (hx[3], cx[3]))
            hx4, cx4 = self.conv_lstm5(hx3, (hx[4], cx[4]))
           #hx5, cx5 = self.conv_lstm6(hx4, (hx[5], cx[5]))
           #hx6, cx6 = self.conv_lstm7(hx5, (hx[6], cx[6]))
           #hx7, cx7 = self.conv_lstm8(hx6, (hx[7], cx[7]))
           #hx8, cx8 = self.conv_lstm9(hx7, (hx[8], cx[8]))
           #hx9, cx9 = self.conv_lstm10(hx8, (hx[9], cx[9]))
            x = hx4
            x = torch.tanh(self.conv_out(x))
            x = torch.tanh(self.conv_1(x))
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.linear_out_0(x))
            x = torch.tanh(self.linear_out_1(x))
           #action = self.actor_linear(x)
           #x = hx4
            value = self.critic_linear(x)

            action = self.actor_linear(x)
            
           #hx10, cx10 = self.actor_conv(x, (hx[10], cx[10]))
           #action = hx10
           #action = action.view(action.size(0), -1)
           #hx11, cx11 = self.critic_conv(x, (hx[11], cx[11]))
           #value = hx11
           #value = value.view(value.size(0), -1)
            return value, action, ([
                                 hx0,
                                 hx1, 
                                 hx2, hx3, hx4,
                                #hx5, hx6, hx7, hx8, hx9,
                                #hx10, hx11

                                ], 
                                [ 
                                 cx0, 
                                 cx1, 
                                 cx2, cx3, cx4,
                                #cx5, cx6, cx7, cx8, cx9,
                                #cx10, cx11
                                ])

class A3Cmicropolis20x20linAC(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=20, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis20x20linAC, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        self.conv_in = nn.Conv2d(14, 16, 1, stride=1, padding=0)                  
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_lstm5 = ConvLSTMCell(16, 16)
        self.conv_lstm6 = ConvLSTMCell(16, 16)


        self.conv_out = nn.Conv2d(16, 8, 3, stride=1, padding=1)                        
        self.conv_1 = nn.Conv2d(8, 16, 3, 1, 1) # 9*9*6 = 1944
        self.conv_2 = nn.Conv2d(16, 16, 3, 3, 1)  # 16*16*6 = 1536

        self.linear_out_0 = nn.Linear(784, 512)
        
        self.linear_out_1 = nn.Linear(512, 1536)

        self.critic_linear = nn.Linear(1536, 1)
        self.actor_linear = nn.Linear(1536, 1600)
        self.postaction_0 = nn.Linear(400, 800)
        
       #self.critic_conv = nn.Conv2d(16, 1, map_width, padding=0)
      # self.critic_linear = nn.Linear(800, 1)
    #self.actor_conv = nn.Conv2d(16, 8, 21, stride=1, padding=10)
       #self.actor_linear = nn.Linear(1600, 800)

        ###########################################################

        self.apply(weights_init)
       #relu_gain = nn.init.calculate_gain('relu')
       #self.conv_in.weight.data.mul_(relu_gain)
       #self.conv_lstm1.weight.data.mul_(relu_gain)
       #self.conv_lstm2.weight.data.mul_(relu_gain)
       #self.conv_lstm3.weight.data.mul_(relu_gain)
       #self.conv_lstm4.weight.data.mul_(relu_gain)
       #self.conv_lstm5.weight.data.mul_(relu_gain)
       #self.conv_lstm6.weight.data.mul_(relu_gain)
       #self.conv_lstm7.weight.data.mul_(relu_gain)
       #self.conv_lstm8.weight.data.mul_(relu_gain)
       #self.conv_lstm9.weight.data.mul_(relu_gain)
       #self.conv_lstm10.weight.data.mul_(relu_gain)
       #self.conv_out.weight.data.mul_(relu_gain)
       #self.actor_conv.weight.data.mul_(relu_gain)
       #self.critic_conv.weight.data.mul_(relu_gain)


       #self.actor_linear.weight.data = norm_col_init(
       #    self.actor_linear.weight.data, 0.01)
       #self.actor_linear.bias.data.fill_(0)
       #self.critic_linear.weight.data = norm_col_init(
       #    self.critic_linear.weight.data, 1.0)
       #self.critic_linear.bias.data.fill_(0)

       #self.lin_lstm.bias_ih.data.fill_(0)
       #self.lin_lstm.bias_hh.data.fill_(0)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 20, 20) 
                for i in range(6)] # + [(1, 8, 10, 10) for i in range(2)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[4], cx[4]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[3], cx[3]))
            hx2, cx2 = self.conv_lstm3(hx1, (hx[2], cx[2]))
            hx3, cx3 = self.conv_lstm4(hx2, (hx[1], cx[1]))
            hx4, cx4 = self.conv_lstm5(hx3, (hx[0], cx[0]))
            hx5, cx5 = self.conv_lstm6(hx4, (hx[5], cx[5]))
           #hx6, cx6 = self.conv_lstm7(hx5, (hx[6], cx[6]))
           #hx7, cx7 = self.conv_lstm8(hx6, (hx[7], cx[7]))
           #hx8, cx8 = self.conv_lstm9(hx7, (hx[8], cx[8]))
           #hx9, cx9 = self.conv_lstm10(hx8, (hx[9], cx[9]))
            x = hx5
            x = torch.tanh(self.conv_out(x))
            x = torch.tanh(self.conv_1(x))
            x = torch.tanh(self.conv_2(x))
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.linear_out_0(x))
            x = torch.tanh(self.linear_out_1(x))
           #action = self.actor_linear(x)
           #x = hx4
            value = self.critic_linear(x)
            action = torch.tanh(self.actor_linear(x))
            action = action.view(action.size(0), 4, 400)
            action = self.postaction_0(action)
         #  action = action.view(400, )
          # action = self.postaction_1(action)
            action = action.view(1, -1)
           #hx10, cx10 = self.actor_conv(x, (hx[10], cx[10]))
           #action = hx10
           #action = action.view(action.size(0), -1)
           #hx11, cx11 = self.critic_conv(x, (hx[11], cx[11]))
           #value = hx11
           #value = value.view(value.size(0), -1)
            return value, action, ([
                                 hx0,
                                 hx1, 
                                 hx2, hx3, hx4,
                                 hx5, 
                                #hx6, hx7, hx8, hx9,
                                #hx10, hx11

                                ], 
                                [ 
                                 cx0, 
                                 cx1, 
                                 cx2, cx3, cx4,
                                 cx5, 
                                #cx6, cx7, cx8, cx9,
                                #cx10, cx11
                                ])
                                
                                
class A3Cmicropolis40x40linAC(torch.nn.Module):

    def __init__(self, num_inputs, action_space, 
             map_width=40, num_obs_channels=14, num_tools=8):

        super(A3Cmicropolis40x40linAC, self).__init__()

        from ConvLSTMCell import ConvLSTMCell

        self.conv_in = nn.Conv2d(14, 16, 1, stride=1, padding=0)                  
        self.conv_lstm1 = ConvLSTMCell(16, 16)
        self.conv_lstm2 = ConvLSTMCell(16, 16)
        self.conv_lstm3 = ConvLSTMCell(16, 16)
        self.conv_lstm4 = ConvLSTMCell(16, 16)
        self.conv_lstm5 = ConvLSTMCell(16, 16)
        self.conv_lstm6 = ConvLSTMCell(16, 16)
        self.conv_lstm7 = ConvLSTMCell(16, 16)
        self.conv_lstm8 = ConvLSTMCell(16, 16)
        self.conv_lstm9 = ConvLSTMCell(16, 16)
        self.conv_lstm10 = ConvLSTMCell(16, 16)

        self.conv_out = nn.Conv2d(16, 8, 3, stride=1, padding=0)                        
        self.conv_1 = nn.Conv2d(8, 6, 3, 1, 0) # 36*36 
        self.conv_2 = nn.Conv2d(6, 6, 3, 1, 0)  # 34*34
        self.conv_3 = nn.Conv2d(6, 6, 3, 2, 0)  # 32*32

        self.linear_out_0 = nn.Linear(1536, 1024)

       #self.lin_lstm = nn.LSTMCell(1024, 1024)
       #self.linear_out_1 = nn.Linear(1024, 1024)

        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, 1600)
        self.postaction_0 = nn.Linear(400, 800)        
        self.postaction_1 = nn.Linear(400, 1600)
        
        ###########################################################

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv_in.weight.data.mul_(relu_gain)

        self.train()

    def getMemorySizes(self):
        return [(1, 16, 40, 40) for i in range(10)]
        #+ [(1, 8, 10, 10) for i in range(2)]


    def forward(self, inputs):
            
        
            inputs, (hx, cx) = inputs
            
            x = inputs
            x = F.relu(self.conv_in(x))
            hx0, cx0 = self.conv_lstm1(x, (hx[9], cx[9]))
            hx1, cx1 = self.conv_lstm2(hx0, (hx[8], cx[8]))
            hx2, cx2 = self.conv_lstm3(hx1, (hx[7], cx[7]))
            hx3, cx3 = self.conv_lstm4(hx2, (hx[6], cx[6]))
            hx4, cx4 = self.conv_lstm5(hx3, (hx[5], cx[5]))
            hx5, cx5 = self.conv_lstm6(hx4, (hx[4], cx[4]))
            hx6, cx6 = self.conv_lstm7(hx5, (hx[3], cx[3]))
            hx7, cx7 = self.conv_lstm8(hx6, (hx[2], cx[2]))
            hx8, cx8 = self.conv_lstm9(hx7, (hx[1], cx[1]))
            hx9, cx9 = self.conv_lstm10(hx8, (hx[0], cx[0]))
            x = hx9
            x = torch.tanh(self.conv_out(x))
            x = torch.tanh(self.conv_1(x))
            x = torch.tanh(self.conv_2(x))
            x = torch.tanh(self.conv_3(x))
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.linear_out_0(x))
           #hx10, cx10 = self.lin_lstm(x, (hx[10], cx[10]))
           #x = hx10
           #x = torch.tanh(self.linear_out_1(x))
           #action = self.actor_linear(x)
           #x = hx4
            value = self.critic_linear(x)
            action = torch.tanh(self.actor_linear(x))
            action = action.view(action.size(0), 4, 400)
            action = torch.tanh(self.postaction_0(action))
            action = action.view(action.size(0), 8, 400)
            action = self.postaction_1(action)
         #  action = action.view(400, )
          # action = self.postaction_1(action)
            action = action.view(1, -1)
           #action[0, 2400:] = torch.full(size=(1, 10400), fill_value=-10)
           #hx10, cx10 = self.actor_conv(x, (hx[10], cx[10]))
           #action = hx10
           #action = action.view(action.size(0), -1)
           #hx11, cx11 = self.critic_conv(x, (hx[11], cx[11]))
           #value = hx11
           #value = value.view(value.size(0), -1)
            return value, action, ([
                                 hx0,
                                 hx1, 
                                 hx2, hx3, hx4,
                                 hx5, 
                                 hx6, hx7, hx8, hx9,
                                #hx10
                                ], 
                                [ 
                                 cx0, 
                                 cx1, 
                                 cx2, cx3, cx4,
                                 cx5, 
                                 cx6, cx7, cx8, cx9,
                                #cx10
                                ])
