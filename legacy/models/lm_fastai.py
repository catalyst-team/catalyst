import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class RnnEncoder(nn.Module):
    """
    A custom RNN encoder network that uses
    - an embedding matrix to encode input,
    - a stack of LSTM layers to drive the network, and
    - variational dropouts in the embedding and LSTM layers

    The architecture for this network was inspired by the work done in
    "Regularizing and Optimizing LSTM Language Models".
    (https://arxiv.org/pdf/1708.02182.pdf)

    source: fast.ai, https://arxiv.org/abs/1801.06146
    """

    initrange = 0.1

    def __init__(
            self, ntoken, emb_sz, nhid, nlayers, pad_token, bidir=False,
            dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5):
        """ Default constructor for the RNN_Encoder class

            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens)
                    in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                nhid (int): number of hidden activation per LSTM layer
                nlayers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations
                    going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM"s internal (or hidden)
                    recurrent weights.

            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [
            nn.LSTM(
                emb_sz if layer == 0 else nhid,
                (nhid if layer != nlayers - 1 else emb_sz) // self.ndir,
                1, bidirectional=bidir)
            for layer in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz, self.nhid, self.nlayers, self.dropoute = (
            emb_sz, nhid, nlayers, dropoute)
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([
            LockedDropout(dropouth) for l in range(nlayers)])

    def forward(self, input, hidden):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)):
                list of tensors evaluated from each RNN layer without using
                dropouth,
                list of tensors evaluated from each RNN layer using dropouth,
        """
        with torch.set_grad_enabled(self.training):
            emb = self.encoder_with_dropout(
                input,
                dropout=self.dropoute if self.training else 0)
            emb = self.dropouti(emb)
            raw_output = emb
            new_hidden, raw_outputs, outputs = [], [], []
            for layer, (rnn, drop) in \
                    enumerate(zip(self.rnns, self.dropouths)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, hidden[layer])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if layer != self.nlayers - 1:
                    raw_output = drop(raw_output)
                outputs.append(raw_output)

            # hidden = repackage_var(new_hidden)
            hidden = new_hidden
        return (raw_outputs, outputs), hidden

    def one_hidden(self, layer, bs):
        weights = next(self.parameters()).data
        nh = (self.nhid
              if layer != self.nlayers - 1
              else self.emb_sz) // self.ndir
        return Variable(weights.new(self.ndir, bs, nh).zero_())

    def init_hidden(self, bs):
        hidden = [
            (self.one_hidden(l, bs), self.one_hidden(l, bs))
            for l in range(self.nlayers)]
        return hidden


class LinearDecoder(nn.Module):
    initrange = 0.1

    def __init__(self, n_out, nhid, dropout, tie_encoder=None, bias=False):
        super().__init__()
        self.decoder = nn.Linear(nhid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


class EmbeddingDropout(nn.Module):
    """
    Applies dropout in the embedding layer by zeroing out
        some elements of the embedding vector.
    Uses the dropout_mask custom layer to achieve this.

    Args:
        embed (torch.nn.Embedding): An embedding torch layer
        words (torch.nn.Variable): A torch variable
        dropout (float): dropout fraction to apply to the embedding weights
        scale (float): additional scaling to apply
            to the modified embedding weights

    Returns:
        tensor of size: (batch_size x seq_length x embedding_size)

    Example:

    >> embed = torch.nn.Embedding(10,3)
    >> words = Variable(torch.LongTensor([[1,2,4,5] ,[4,3,2,9]]))
    >> words.size()
        (2,4)
    >> embed_dropout_layer = EmbeddingDropout(embed)
    >> dropout_out_ = embed_dropout_layer(embed, words, dropout=0.40)
    >> dropout_out_
        Variable containing:
        (0 ,.,.) =
          1.2549  1.8230  1.9367
          0.0000 -0.0000  0.0000
          2.2540 -0.1299  1.5448
          0.0000 -0.0000 -0.0000

        (1 ,.,.) =
          2.2540 -0.1299  1.5448
         -4.0457  2.4815 -0.2897
          0.0000 -0.0000  0.0000
          1.8796 -0.4022  3.8773
        [torch.FloatTensor of size 2x4x3]
    """

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            size = (self.embed.weight.size(0), 1)
            mask = Variable(dropout_mask(
                self.embed.weight.data, size, dropout))
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight

        if scale:
            masked_embed_weight = scale * masked_embed_weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = F.embedding(
            words,
            masked_embed_weight, padding_idx, self.embed.max_norm,
            self.embed.norm_type, self.embed.scale_grad_by_freq,
            self.embed.sparse)

        return X


class WeightDrop(torch.nn.Module):
    """A custom torch layer that serves as a wrapper on another torch layer.
    Primarily responsible for updating the weights in the wrapped module based
    on a specified dropout.
    """

    def __init__(self, module, dropout, weights=["weight_hh_l0"]):
        """ Default constructor for the WeightDrop module

        Args:
            module (torch.nn.Module): A pytorch layer being wrapped
            dropout (float): a dropout value to apply
            weights (list(str)): the parameters of the wrapped **module**
                which should be fractionally dropped.
        """
        super().__init__()
        self.module, self.weights, self.dropout = module, weights, dropout
        self._setup()

    def _setup(self):
        """ for each string defined in self.weights, the corresponding
        attribute in the wrapped module is referenced,
        then deleted, and subsequently
        registered as a new parameter with a slightly modified name.

        Args:
            None

         Returns:
             None
        """
        if isinstance(self.module, torch.nn.RNNBase):
            self.module.flatten_parameters = noop
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            w.required_grad = False  # bugfix
            # del self.module._parameters[name_w]  # bugfix
            self.module.register_parameter(
                name_w + "_raw", nn.Parameter(w.data))

    def _setweights(self):
        """
        Uses pytorch"s built-in dropout function to apply dropout
            to the parameters of the wrapped module.
        Args:
            None
        Returns:
            None
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = torch.nn.functional.dropout(
                raw_w, p=self.dropout, training=self.training)
            if hasattr(self.module, name_w):
                delattr(self.module, name_w)
            setattr(self.module, name_w, w)
            self.module._parameters[name_w] = w  # bugfix

    def forward(self, *args):
        """
        updates weights and delegates the propagation
            of the tensor to the wrapped module"s forward method
        Args:
            *args: supplied arguments
        Returns:
            tensor obtained by running the forward method on the wrapped module
        """
        self._setweights()
        return self.module.forward(*args)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(m, requires_grad=False) * x


class LmModel(nn.Module):
    def __init__(
            self, n_tok, emb_sz, nhid, nlayers, pad_token,
            dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5,
            tie_weights=True, bias=False):
        """Returns a SequentialRNN model.

            A RNN_Encoder layer is instantiated using the parameters provided.

            This is followed by the creation of a LinearDecoder layer.

            Also by default (i.e. tie_weights = True),
            the embedding matrix used in the RNN_Encoder
            is used to  instantiate the weights for the LinearDecoder layer.

            The SequentialRNN layer is the native torch"s Sequential wrapper
                that puts the RNN_Encoder and
            LinearDecoder layers sequentially in the model.

            Args:
                n_tok (int): number of unique vocabulary words (or tokens)
                    in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                nhid (int): number of hidden activation per LSTM layer
                nlayers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations
                    going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM"s
                    internal (or hidden) recurrent weights.
                tie_weights (bool): decide if the weights
                    of the embedding matrix
                    in the RNN encoder should be tied to the
                    weights of the LinearDecoder layer.
                bias (bool): decide if the decoder should have
                    a bias layer or not.
            """
        super().__init__()
        self.rnn_enc = RnnEncoder(
            n_tok, emb_sz, nhid=nhid, nlayers=nlayers, pad_token=pad_token,
            dropouth=dropouth, dropouti=dropouti,
            dropoute=dropoute, wdrop=wdrop)
        enc = self.rnn_enc.encoder if tie_weights else None
        self.decoder = LinearDecoder(
            n_tok, emb_sz, dropout,
            tie_encoder=enc, bias=bias)

    def forward(self, input, hidden):
        output, hidden = self.rnn_enc(input, hidden)
        output = self.decoder(output)
        return tuple(output), hidden


def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument "sz".
    Args:
        x (nn.Variable): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution
        to decide which activations to keep.
    Additionally, the sampled activations
        is rescaled is using the factor 1/(1 - dropout).

    In the example given below,
        one can see that approximately .8 fraction of the
        returned tensors are zero.
    Rescaling with the factor 1/(1 - 0.8) returns a tensor
        with 5"s in the unit places.

    The official link to the pytorch bernoulli function is here:
        http://pytorch.org/docs/master/torch.html#torch.bernoulli

    Examples:
        >>> a_Var = torch.Tensor(2, 3, 4).uniform_(0, 1).detach()
        >>> a_Var
            Variable containing:
            (0 ,.,.) =
              0.6890  0.5412  0.4303  0.8918
              0.3871  0.7944  0.0791  0.5979
              0.4575  0.7036  0.6186  0.7217
            (1 ,.,.) =
              0.8354  0.1690  0.1734  0.8099
              0.6002  0.2602  0.7907  0.4446
              0.5877  0.7464  0.4257  0.3386
            [torch.FloatTensor of size 2x3x4]
        >>> a_mask = dropout_mask(
                a_Var.data, (1,a_Var.size(1),a_Var.size(2)), dropout=0.8)
        >>> a_mask
            (0 ,.,.) =
              0  5  0  0
              0  0  0  5
              5  0  5  0
            [torch.FloatTensor of size 1x3x4]
    """
    return x.new(*sz).bernoulli_(1 - dropout) / (1 - dropout)


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return (
        h.detach()
        if type(h) == torch.Tensor
        else tuple(repackage_var(v) for v in h))


def noop(*args, **kwargs): return
