from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSNet(nn.Module):

    def __init__(self, args, n_token, dictionary, app_list):
        super(FSNet, self).__init__()
        self.batch_size = args.batch_size
        self.hidden_unit = args.hidden_unit
        self.n_token = n_token
        self.dictionary = dictionary
        self.class_num = len(app_list)
        self.use_cuda = args.cuda
        self.max_length = args.max_length

        self.embedding = nn.Embedding(self.n_token, args.embedding_size)
        self.encoder_gru = nn.GRU(args.embedding_size, self.hidden_unit, args.layers_num, dropout=args.dropout, bidirectional=True, batch_first=True)
        self.decoder_gru = nn.GRU(4 * self.hidden_unit, self.hidden_unit, args.layers_num, dropout=args.dropout, bidirectional=True, batch_first=True)
        self.reconstruction_nn = nn.Linear(4 * self.hidden_unit, self.class_num)

        self.dense_net_1 = nn.Linear(16 * self.hidden_unit, 16 * self.hidden_unit)
        self.dense_net_2 = nn.Linear(16 * self.hidden_unit, 16 * self.hidden_unit)
        self.dense_net_3 = nn.Linear(16 * self.hidden_unit, self.class_num)

        self.drop = nn.Dropout(args.dropout)
        self.activation = nn.ReLU()
        # self.init_weights()

    def init_weights(self, init_range=0.1):
        # self.fc.weight.data.uniform_(-init_range, init_range)
        # self.fc.bias.data.fill_(0)
        # self.pred.weight.data.uniform_(-init_range, init_range)
        # self.pred.bias.data.fill_(0)
        pass

    def forward(self, input_x):
        # embedding
        embedding_x = self.embedding(input_x)

        # init hidden
        batch_size = input_x.shape[0]
        encoder_hidden = torch.zeros(4, batch_size, self.hidden_unit)
        decoder_hidden = torch.zeros(4, batch_size, self.hidden_unit)
        if self.use_cuda:
            encoder_hidden = encoder_hidden.cuda()
            decoder_hidden = decoder_hidden.cuda()

        # encoder
        _, encoder_output = self.encoder_gru(embedding_x, encoder_hidden)
        encoder_output = encoder_output.transpose(0, 1)
        z_e = encoder_output.reshape((encoder_output.shape[0], -1))
        z_e_repeat = z_e.repeat(self.max_length, 1, 1).transpose(0, 1)

        # decoder
        _, decoder_output = self.decoder_gru(z_e_repeat, decoder_hidden)
        decoder_output = decoder_output.transpose(0, 1)
        z_d = decoder_output.reshape((decoder_output.shape[0], -1))

        # reconstruction layer
        recon = self.reconstruction_nn(z_e)

        # dense net
        dense_net_input = torch.cat([z_e, z_d, z_e*z_d, torch.abs(z_e - z_d)],dim=-1)
        dense_output_1 = self.drop(self.activation(self.dense_net_1(dense_net_input)))
        dense_output_2 = self.drop(self.activation(self.dense_net_2(dense_output_1)))
        dense_output_3 = self.dense_net_3(dense_output_2)

        return recon, dense_output_3
