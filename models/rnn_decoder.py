import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class DecoderRNN(nn.Module):
    """
    source: PyTorch tutorials
    """
    def __init__(
            self,
            vocabulary_size, embedding_size,
            hidden_size, num_layers,
            input_size=None,
            input_usage="first_token",
            **kwargs):
        """

            input_usage: "first_token" or "hidden_state"
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocabulary_size)
        self.init_weights()

        self.input_usage = input_usage

        if input_usage == "first_token":
            # @TODO: how to replace tanh with something better?
            self.input_process = (
                nn.Sequential(
                    nn.Linear(input_size, embedding_size),
                    nn.Tanh())
                if input_size is not None
                else None)
            self.forward_fn = self.forward_with_first_token
            self.sample_fn = self.sample_with_first_token
        elif input_usage == "hidden_state":
            self.input_process = (
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.Tanh())
                    for _ in range(num_layers)])
                if input_size is not None
                else None)
            self.forward_fn = self.forward_with_hidden_state
            self.sample_fn = self.sample_with_hidden_state
        else:
            raise Exception("unknown image usage")

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward_with_first_token(self, features, captions, lengths):
        if self.input_process is not None:
            features = self.input_process(features)
        features = features.unsqueeze(1)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features, embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, _ = self.lstm(packed)
        output = self.linear(output[0])  # take input from PackedSequence
        return output

    def forward_with_hidden_state(self, features, captions, lengths):
        embeddings = self.embed(captions)
        if self.input_process is not None:
            features = [layer(features) for layer in self.input_process]
        hx = torch.cat(features, 0)
        if len(hx.shape) < 3:
            hx = hx.unsqueeze(0)
        cx = hx.new_zeros(hx.shape)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, _ = self.lstm(packed, (hx, cx))
        output = self.linear(output[0])  # take input from PackedSequence
        return output

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        return self.forward_fn(features, captions, lengths)

    def sample_with_first_token(
            self, features, states=None, suffix=None, max_len=50):
        """
        suffix unused
        """
        sampled_ids = []
        if self.input_process is not None:
            features = self.input_process(features)
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # (bs, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # (bs, vocab_size)
            predicted = outputs.max(1)[1]  # argmax
            sampled_ids.append(predicted.unsqueeze(1))
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)  # (batch_size, max_len)
        return sampled_ids

    def sample_with_hidden_state(
            self, features, states=None, suffix=None, max_len=50):
        """
        states unused
        """
        sampled_ids = []
        inputs = self.embed(suffix)
        if self.input_process is not None:
            features = [layer(features) for layer in self.input_process]
        hx = torch.cat(features, 0)
        if len(hx.shape) < 3:
            hx = hx.unsqueeze(0)  # bugfix for 1-layer lstm
        cx = hx.new_zeros(hx.shape)
        states = (hx, cx)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # (bs, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # (bs, vocab_size)
            predicted = outputs.max(1)[1]                # argmax
            sampled_ids.append(predicted.unsqueeze(1))
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                 # (bs, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)          # (bs, max_len)
        return sampled_ids

    def sample(self, features, states=None, suffix=None, max_len=50):
        """Samples captions for given image features (Greedy search)."""
        return self.sample_fn(features, states, suffix, max_len)
