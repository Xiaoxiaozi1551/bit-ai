import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import gamma, device, batch_size

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively. 
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers. 
        # This function now only implements two fully connected layers. Modify this to include LSTM layer(s). 
        
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(self.num_inputs, 128)
        self.lstm = nn.LSTM(128, 16)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)



    def forward(self, x, hidden=None):
        # The variable x denotes the input to the network. 
        # The hidden variable denotes the hidden state and cell state inputs to the LSTM based network. 
        # The function returns the q value and the output hidden variable information (new cell state and new hidden state) for the given input. 
        # This function now only uses the fully connected layers. Modify this to use the LSTM layer(s).          
        if hidden is None:
            # 如果隐藏状态为空，则初始化为全零张量
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)
        # print(x.size(), '1')
        # print(hidden, 'hidden')
        out = F.relu(self.fc1(x))
        # print(out.size(), '2')
        # out = out.view(1, 1, -1)  # Reshape output to (1, 1, 16) for LSTM input
        # print(out.size(), '3')
        # out, hidden = self.lstm(out, hidden)
        # print(out.size(), '4')
        # out = out.view(-1, 16)  # Reshape output to (1, 16) for fully connected layer
        # print(out.size(), '5')
        qvalue = self.fc2(out)

        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        # The online_net is the variable that represents the first (current) Q network.
        # The target_net is the variable that represents the second (target) Q network.
        # The optimizer is Adam. 
        # Batch represents a mini-batch of memory. Note that the minibatch also includes the rnn state (hidden state) for the DRQN. 

        # This function takes in a mini-batch of memory, calculates the loss and trains the online network. Target network is not trained using back prop. 
        # The loss function is the mean squared TD error that takes the difference between the current q and the target q. 
        # Return the value of loss for logging purposes (optional).

        # Implement this function. Currently, temporary values to ensure that the program compiles.
        states, next_states, actions, rewards, masks, hidden = batch

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        masks = torch.cat(masks)

        q_values, _ = online_net(states, hidden)
        next_q_values, _ = target_net(next_states, hidden)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0].detach()

        expected_q_value = rewards + gamma * next_q_value * masks

        loss = F.smooth_l1_loss(q_value, expected_q_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()



    def get_action(self, state, hidden):
        # state represents the state variable. 
        # hidden represents the hidden state and cell state for the LSTM.
        # This function obtains the action from the DRQN. The q value needs to be obtained from the forward function and then a max needs to be computed to obtain the action from the Q values. 
        # Implement this function. 
        # Template code just returning a random action.

        # state = torch.Tensor(state).unsqueeze(0)
        q_values, hidden = self.forward(state, hidden)
        _, action = torch.max(q_values, 1)
        return action, hidden
