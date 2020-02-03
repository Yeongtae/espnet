import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import math
import numpy as np
import logging
import builtins

def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states

class VGG2L(torch.nn.Module):

    def __init__(self, in_channel=1, vgg_channel=[64,64,128,128]):
        super(VGG2L, self).__init__()

        #self.conv1_1 = torch.jit.trace(torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1), torch.rand(1,1,258,83))
        self.conv1_1 = torch.nn.Conv2d(in_channel, vgg_channel[0], 3, stride=1, padding=1)# torch.rand(1,1,258,83))
        #self.conv1_2 = torch.jit.trace(torch.nn.Conv2d(64, 64, 3, stride=1, padding=1), torch.rand(1,64,258,83))
        #self.conv2_1 = torch.jit.trace(torch.nn.Conv2d(64, 128, 3, stride=1, padding=1), torch.rand(1,64,129,42))
        #self.conv2_2 = torch.jit.trace(torch.nn.Conv2d(128, 128, 3, stride=1, padding=1), torch.rand(1,128,129,32))
        self.conv1_2 = torch.nn.Conv2d(vgg_channel[0], vgg_channel[1], 3, stride=1, padding=1)# torch.rand(1,64,258,83))
        self.conv2_1 = torch.nn.Conv2d(vgg_channel[1], vgg_channel[2], 3, stride=1, padding=1)# torch.rand(1,64,129,42))
        self.conv2_2 = torch.nn.Conv2d(vgg_channel[2], vgg_channel[3], 3, stride=1, padding=1)# torch.rand(1,128,129,32))
        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel, xs_pad.size(2) // self.in_channel).transpose(1, 2)
        #print('VGG2L-before view, xs_pad.size={}, ilens={}'.format(xs_pad.size(), ilens))
        xs_pad = F.relu(self.conv1_1(xs_pad))
        #print('VGG2L-1relu, xs_pad.size={}, ilens={}'.format(xs_pad.size(), ilens))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        #print('VGG2L-2relu, xs_pad.size={}, ilens={}'.format(xs_pad.size(), ilens))
        xs_pad = torch.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        #print('VGG2L-1maxpool, xs_pad.size={}, ilens={}'.format(xs_pad.size(), ilens))

        xs_pad = F.relu(self.conv2_1(xs_pad))
        #print('VGG2L-3relu, xs_pad.size={}, ilens={}'.format(xs_pad.size(), ilens))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        #print('VGG2L-4relu, xs_pad.size={}, ilens={}'.format(xs_pad.size(), ilens))
        xs_pad = torch.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        ilens = torch.ceil(ilens.to(torch.float32)/2).to(torch.int64)
        ilens = torch.ceil(ilens.to(torch.float32)/2).to(torch.int64)

        xs_pad = torch.transpose(xs_pad, 1, 2)
        xs_pad = xs_pad.contiguous().view(xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        #print("VGG2L forward xs_pad.size()={}, ilens={}".format(xs_pad.size(), ilens))
        return xs_pad, ilens, torch.rand([0])

class RNN(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = True
        self.nbrnn = torch.nn.LSTM(
            idim, cdim, elayers, batch_first=True, dropout=0, bidirectional=bidir)

        self.l_last = torch.nn.Linear(cdim * 2, hdim)
        self.typ = typ

    def forward(self, xs_pad, ilens):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        #print("RNN, xs_pad.size={}, ilens={}, prev_state={}".format(xs_pad.size(),ilens, prev_state))

        # if torch.equal(prev_state, torch.tensor(0)):
        # prev_state=None
        #print("encoder before pack_padded={}".format(ilens))
        #xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        #self.nbrnn.flatten_parameters()
        #if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
        #prev_state = reset_backward_rnn_state(prev_state)
        #print('xs_pack={}'.format(xs_pack))
        #input, batch_sizes, sorted_indices, unsorted_indices = xs_pack
        #print('input={}, batch_sizes={}'.format(input, batch_sizes))i
        #xs_pack.batch_sizes = None
        ys_pad, states = self.nbrnn(xs_pad, hx=None)

        #print("RNN states={}".format(states))
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        #ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)

        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        #print("RNN, xs_pad.size={}, ilens={}, prev_state={}".format(xs_pad.size(),ilens, states))
        return xs_pad, ilens, states  # x: utt list of frame x dim


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1, vgg_channel=[64,64,128,128]):
        super(Encoder, self).__init__()
        typ = 'blstm'
        self.enc = torch.nn.ModuleList([VGG2L(in_channel, vgg_channel),
                                        RNN(get_vgg2l_odim(idim, in_channel=in_channel, out_channel=vgg_channel[-1]), elayers, eunits,
                                            eprojs,
                                            dropout, typ=typ)])

        #self.vgg2l = VGG2L(in_channel)
        #self.rnn = RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)

    def forward(self, xs_pad, ilens):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        #if prev_states is None:
        prev_states = [None] * 2
        #assert len(prev_states) == len(self.enc)

        for module in self.enc:
            xs_pad, ilens, states = module(xs_pad, ilens)

        return xs_pad, ilens


def get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = torch.ceil(torch.tensor(
        idim, dtype=torch.float32) / 2)  # 1st max pooling
    idim = torch.ceil(torch.tensor(
        idim, dtype=torch.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    #assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)


@torch.jit.script
def make_pad_mask(lengths):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    #length_dim = int(length_dim)
    # if length_dim == 0:
    #    raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    # if not isinstance(lengths, list):
    #lengths = lengths.tolist()

    lsize = list(lengths.size())
    bs = int(len(lsize))
    print("bsbsbssbsbsbs={}".format(bs))
#    if xs is None:
    #maxlen = int(max(lengths))
    maxlen = int(torch.max(lengths).item())
    print("lengths...............={}".format(maxlen))
#    else:
#        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    #seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    print("mask-------------------------={}".format(mask))
    """
    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)

        #ind = tuple(slice(None) if i in (0, length_dim) else None
        #            for i in range(xs.dim()))

        ind = []
        for i in range(xs.dim()):
            #temp = ()

            if i in [0, length_dim]:
                temp = slice(None)
            else:
                temp = None
            ind.append(temp)
        ind = tuple(ind)

        mask = mask[ind].expand_as(xs).to(xs.device) """
    return mask



class DecoderRNNT(torch.nn.Module):
    """RNN-T Decoder module.

    Args:
        eprojs (int): # encoder projection units
        odim (int): dimension of outputs
        dtype (str): gru or lstm
        dlayers (int): # prediction layers
        dunits (int): # prediction units
        blank (int): blank symbol id
        embed_dim (init): dimension of embeddings
        joint_dim (int): dimension of joint space
        dropout (float): dropout rate
        dropout_embed (float): embedding dropout rate
        rnnt_type (str): type of rnn-t implementation

    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, blank,
                 embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0,
                 rnnt_type='warp-transducer'):
        """Transducer initializer."""
        super(DecoderRNNT, self).__init__()

        self.embed = torch.nn.Embedding(odim, embed_dim, padding_idx=blank)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        dec_net = torch.nn.LSTMCell

        self.decoder = torch.nn.ModuleList([dec_net(embed_dim, dunits)])
        self.dropout_dec = torch.nn.ModuleList([torch.nn.Dropout(p=dropout)])

        for _ in range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

        # from warprnnt_pytorch import RNNTLoss
        # self.rnnt_loss = RNNTLoss(blank=blank)

        self.lin_enc = torch.nn.Linear(eprojs, joint_dim)
        self.lin_dec = torch.nn.Linear(dunits, joint_dim, bias=False)
        self.lin_out = torch.nn.Linear(joint_dim, odim)

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.odim = odim

        self.rnnt_type = rnnt_type

        self.ignore_id = -1
        self.blank = blank
        self.test = True

    # @torch.jit.script
    def zero_state(self, ey, dunits: int):
        """Initialize decoder states.

        Args:
            ey (torch.Tensor): batch of input features (B, Emb_dim)

        Returns:
            (list): list of L zero-init hidden and cell state (B, Hdec)

        """
        #z_list = ey.new_zeros(ey.size(0), self.dunits)
        #c_list = ey.new_zeros(ey.size(0), self.dunits)
        z_list = [torch.zeros(ey.size(0), dunits)]
        c_list = [torch.zeros(ey.size(0), dunits)]

        # for _ in six.moves.range(1, dlayers):
        #    z_list.append(ey.new_zeros(ey.size(0), dunits))
        #    c_list.append(ey.new_zeros(ey.size(0), dunits))

        return (z_list, c_list)

    def call_choice(self, ey, z_prev, c_prev, choice_idx: int):
        for module in self.decoder:
            #print("decoder decoder ={}".format(module))
            #return module(ey, (z_prev, c_prev))
            z_list, c_list = module(ey, (z_prev, c_prev))
            #print("call choice z_list={}, c_list={}".format(z_list, c_list))
            return z_list, c_list

    def call_dropout(self, z_list, choice_idx: int):
        for module in self.dropout_dec:
            return module(z_list)

    # @torch.jit.script
    def rnn_forward(self, ey, dstatel, dstater):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, Emb_dim)
            dstate (list): list of L input hidden and cell state (B, Hdec)

        Returns:
            output (torch.Tensor): batch of output features (B, Hdec)
            dstate (list): list of L output hidden and cell state (B, Hdec)

        """

        if dstatel is None and dstater is None:
            z_prev = self.zero_state(ey, self.dunits)[0]
            c_prev = self.zero_state(ey, self.dunits)[1]
        else:
            z_prev = dstatel
            c_prev = dstater

        z_list, c_list = self.zero_state(ey, self.dunits)
        #c_list = self.zero_state(ey, self.dunits)

        #print("run_forward, z_prev=={}, c_prev=={}".format(z_prev, c_prev))
        #z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
        #print("ey.size={}, ey={}, z_prev.size={}, z_prev={}".format(ey.size(), ey, z_prev.size(), z_prev))
        z_list[0], c_list[0] = self.call_choice(ey, z_prev, c_prev, 0)
        #print("z_list={}".format(z_list))
        for l in range(1, self.dlayers):
            #z_list[l], c_list[l] = self.decoder[l](
            z_list[l], c_list[l] = self.call_choice(
                self.call_dropout(z_list[l - 1], l-1), z_prev[l], c_prev[l], l)

        y = self.call_dropout(z_list[-1], -1)

        return y, (z_list, c_list)

    #@torch.jit.script
    def joint(self, h_enc, h_dec):
        """Joint computation of z.

        Args:
            h_enc (torch.Tensor): batch of expanded hidden state (B, T, 1, Henc)
            h_dec (torch.Tensor): batch of expanded hidden state (B, 1, U, Hdec)

        Returns:
            z (torch.Tensor): output (B, T, U, odim)

        """

        #print('h_enc={}, h_dec={}'.format(h_enc.shape, h_dec.dim()))
        h_enc = h_enc.view(1,list(h_enc.shape)[0])
        h_dec = h_dec.view(1, list(h_dec.shape)[0])
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        z = z.squeeze()
        #print('z value={}, size={}'.format(z, z.shape))
        return z

    def recognize(self, hs_pad, hlens, ys_pad):
        """Forward function for transducer.

        Args:
            hs_pad (torch.Tensor): batch of padded hidden state sequences (B, Tmax, D)
            hlens (torch.Tensor): batch of lengths of hidden state sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
           loss (float): rnnt loss value

        """
        #ys = [y[y != self.ignore_id] for y in ys_pad]

        #for y in ys_pad:
        """
        ys = [ys_pad[0][ys_pad[0] != self.ignore_id]]

        #hlens = list(map(int, hlens))
        #hlens = list(hlens.to(torch.int32))
        hlens = list(hlens.to(torch.int32))
        #blank = ys[0].new([self.blank])
        blank = ys[0].new_full([self.blank], 0)

        #ys_in = [torch.cat([blank, y], dim=0) for y in ys]
        ys_in = [torch.cat([blank, ys[0]], dim=0)]
        #ys_in_pad = pad_list(ys_in, self.blank)
        ys_in_pad = torch.stack(ys_in)

        olength = ys_in_pad.size(1)
        #print("olength={}".format(olength))
        z_list, c_list = self.zero_state(hs_pad, self.dunits)
        eys = self.dropout_embed(self.embed(ys_in_pad))

        print("eys={}, z_list={}, c_list={}".format(eys.size(), z_list, c_list))
        z_all = []
        for i in range(olength):
            y, (z_list, c_list) = self.rnn_forward(eys[:, i, :], z_list, c_list)
            z_all.append(y)

        h_dec = torch.stack(z_all, dim=1)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint(h_enc, h_dec)
        ##y = pad_list(ys, self.blank).type(torch.int32)
        y = torch.stack(ys, dim=0)
        #z_len = to_device(self, torch.IntTensor(hlens))
        #z_len = torch.IntTensor(hlens).to('cpu')
        z_len = torch.tensor(hlens, dtype=torch.int32).to('cpu')
        #y_len = to_device(self, torch.IntTensor([_y.size(0) for _y in ys]))
        #y_len = torch.IntTensor([_y.size(0) for _y in ys]).to('cpu')
        y_len = torch.tensor([ys[0].size(0)], dtype=torch.int32).to('cpu')
        #y_len = ys[0].size(0).to(torch.int32).to('cpu')

        #loss = to_device(self, self.rnnt_loss(z, y, z_len, y_len))
        #loss = self.rnnt_loss(z, y, z_len, y_len)
        """
        return 0.1

    #@torch.jit.script
    def forward(self, h):
        """Greedy search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        z_list, c_list = self.zero_state(h.unsqueeze(0), self.dunits)
        #ey = to_device(self, torch.zeros((1, self.embed_dim)))
        ey = torch.zeros((1, self.embed_dim)).to('cpu')
        temp = float(self.blank)
        hyp = builtins.dict([('score', [0.0]), ('yseq', [temp])])
        print("hyp={}".format(hyp))
        #hyp_score = torch.zeros([0], dtype=torch.float32)
        hyp_score = torch.zeros(0, dtype=torch.float)
        #hyp_score = torch.tensor(0.)
        hyp_yseq = torch.zeros([0], dtype=torch.int32)

        z_list = torch.stack(z_list).squeeze(1)
        c_list = torch.stack(c_list).squeeze(1)
        #print("hyp_yseq.size={}".format(hyp_yseq.size()))
        #print("RNNT recognize ey.size={}, ey={}, z_list={}, c_list={}".format(ey.size(), ey, z_list, c_list))
        y, (z_list, c_list) = self.rnn_forward(ey, z_list, c_list)

        #print("y valu={}".format(y))
        #print("RNNT recognize y.size={}, y={}".format(y.size(), y))
        for hi in h:

            #print('hi value={}, size={}'.format(hi, hi.shape))
            ytu = F.log_softmax(self.joint(hi, y[0]), dim=0)
            #print('ytu={}'.format(ytu))
            ytu = ytu.squeeze()
            logp, pred = torch.max(ytu, dim=0)
            #print('logp={}, pred={}'.format(logp, pred))

           # print("hi.size={}, ytu.size={}, y.size={}, y[0].size={}, logp={}, pred={}".format(hi.size(), ytu.size(), y.size(), y[0].size(), logp, pred))
            #logp_t = torch.tensor([float(logp)], dtype=torch.float32)
            #pred_t = torch.tensor([int(pred)], dtype=torch.int32)
            #print("logp={}, pred={}".format(logp.view(1), hyp_yseq))
            #print('pred.item()={}'.format(pred))
            if pred.item() != self.blank:
                #hyp['yseq'].append(int(pred))
                hyp['score'][0] += float(logp)
                hyp_yseq = torch.cat((hyp_yseq, pred.view(1).to(torch.int32)), dim=0)
                zero_token = torch.ones((1, 1), dtype=torch.long)
                eys = torch.mul(zero_token, hyp_yseq[-1])
                #eys = torch.full((1, 1), hyp_yseq[-1], dtype=torch.long).to('cpu')
                #print("eys print={}, hyp_yseq={}".format(eys, hyp_yseq[-1]))
                ey = self.dropout_embed(self.embed(eys))
                #ey = torch.zeros([1, 256], dtype=torch.float)
                #print("eys print={}, hyp_yseq={}".format(eys, hyp_yseq[-1]))
                #print("recognize, loop, z_list.size={}, c_list.size={}".format(z_list[0], c_list[0]))
                y, (z_list, c_list) = self.rnn_forward(ey[0], z_list[0], c_list[0])
                #print("rnn_forward end")

        return hyp_yseq, hyp['score'][0]


def encoder_for(args, idim: int, subsample):

    return Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)


def decoder_for(args, odim: int, blank=0):
    return DecoderRNNT(args.eprojs, odim, args.dtype, args.dlayers, args.dunits,
                       blank, args.dec_embed_dim, args.joint_dim,
                       args.dropout_rate_decoder, args.dropout_rate_embed_decoder,
                       args.rnnt_type)


class E2E(torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (namespace): argument Namespace containing options

    """

    def __init__(self, idim, odim, args, encoder, decoder):
        """Initialize transducer modules.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.rnnt_mode = args.rnnt_mode
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        #self.reporter = Reporter()
        self.beam_size = args.beam_size

        # note that eos is the same as sos (equivalent ID)
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        self.subsample = np.ones(args.elayers + 1, dtype=np.int)
        self.frontend = None

        #encoder
        #self.enc = encoder_for(args, idim, self.subsample)
        self.enc = encoder

        #prediction
        #self.dec = decoder_for(args, odim)
        self.dec = decoder

        # weight initialization
 #       self.init_like_chainer()

        self.report_cer = False
        self.report_wer = False
        torch.set_printoptions(edgeitems=3)
        self.logzero = -10000000000.0
        self.rnnlm = None
        self.loss = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)

        """
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)

        self.dec.embed.weight.data.normal_(0, 1)

    def forward(self, feat: torch.Tensor, ilens: torch.Tensor):

        #prev = self.training
        #self.eval()
        #ilens = [x.shape[0]]

        # subsample frame
        #x = x[::self.subsample[0], :]
        #h = to_torch_tensor(x).float().to('cpu')
        # make a utt list (1) to use the same interface for encoder
        hs = feat.contiguous().unsqueeze(0)

        # 0. Frontend
        #print("E2E encoder hs size={}".format(hs.size()))
        # 1. Encoder
        h, _ = self.enc(hs, ilens)

        # 2. Decoder
        #print("E2E forward h.size()={}, h[0].size={}".format(h.size(), h[0]))
        yseq, yscore = self.dec(h[0])

        #if prev:
        #    self.train()
        print("yseq.shape={}, yscore={}".format(yseq.shape, yscore))
        return yseq

    def subsample_frames(self, x):
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

