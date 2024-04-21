import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class Self_Attention(nn.Module):
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(Self_Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, key, value):
        # query == hidden: (batch_size, hidden_dim * 2)
        # key/value == gru_output: (sentence_length, batch_size, hidden_dim * 2)
        # query = query.unsqueeze(1) # (batch_size, 1, hidden_dim * 2)
        key = key.transpose(1, 2)  # (batch_size, hidden_dim * 2, sentence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key)  # (batch_size, 1, sentence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2)  # normalize sentence_length's dimension
        # print(attention_weight)
        # value = value.transpose(0, 1) # (batch_size, sentence_length, hidden_dim * 2)
        attention_output = torch.bmm(attention_weight, value)  # (batch_size, 5, 14)
        # attention_output = attention_output.squeeze(1)  # (batch_size, hidden_dim * 2)

        # return attention_output, attention_weight.squeeze(1)
        #return attention_output
        return attention_output, attention_weight

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.bn = nn.BatchNorm1d(5, momentum=0.9)
        self.key = nn.Linear(input_size, input_size)
        self.query = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.attention = Self_Attention(input_size)
        self.GRU_layer = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=False,
                                dropout=0.1)
        self.output_linear = nn.Linear(hidden_size, num_classes)
        # self.softmax = torch.nn.Softmax(dim=1)
        # self.output_linear = nn.Linear(hidden_size * 2, num_classes)
        self.hidden = None

    def forward(self, x):
        # x = self.bn(x)
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        tt, attention_weight = self.attention(query=query, key=key, value=value)
        #x, attention_weight = self.attention(query=query, key=key, value=value)
        # key1 = self.key(x1)
        # query1 = self.query(x1)
        # value1 = self.value(x1)
        # key2 = self.key(x2)
        # query2 = self.query(x2)
        # value2 = self.value(x2)
        # x1, attention_weight1 = self.attention(query=query1, key=key1, value=value1)
        # x2, attention_weight2 = self.attention(query=query2, key=key2, value=value2)
        # x = torch.cat([x1,x2],dim = 0)
        x = x + tt
        # x = self.bn(x)
        x, self.hidden = self.GRU_layer(x)
        x = x[:, -1]
        x = self.output_linear(x)
        # x = self.softmax(x)
        #return x
        return x, attention_weight
        # return output.squeeze(1), attention_weight
#class Model(nn.Module):
#     def __init__(self, input_size, hidden_size,num_layers, num_classes):
#         super(Model, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
#                         bidirectional=True, dropout=d_rate)
#         self.dense = nn.Linear(2 * hidden_dim, output_dim)
#         self.dropout = nn.Dropout(d_rate)
#         self.attention = Self_Attention(2 * hidden_dim)
#
#     def forward(self, x):
#         # x: (sentence_length, batch_size)
#
#         embedded = self.dropout(self.embedding(x))
#         # embedded: (sentence_length, batch_size, embedding_dim)
#
#         gru_output, hidden = self.gru(embedded)
#         # gru_output: (sentence_length, batch_size, hidden_dim * 2)
#         ## depth_wise
#         # hidden: (num_layers * 2, batch_size, hidden_dim)
#         ## ordered: [f_layer_0, b_layer_0, ...f_layer_n, b_layer n]
#
#         # concat the final output of forward direction and backward direction
#         hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
#         # hidden: (batch_size, hidden_dim * 2)
#
#         rescaled_hidden, attention_weight = self.attention(query=hidden, key=gru_output, value=gru_output)
#         output = self.dense(rescaled_hidden)
# from torch.utils.tensorboard import SummaryWriter, writer
# model = GRUModel()
# images = torch.randn(1, 1, 28, 28)
# writer.add_graph(model, images)
# writer.close()
