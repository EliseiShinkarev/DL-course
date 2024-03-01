import torch
from typing import Type
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import TextDataset
from torch.distributions.categorical import Categorical

class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        # self.embedding = None
        # self.rnn = None
        # self.linear = None


        self.embedding = nn.Embedding(num_embeddings=dataset.vocab_size, embedding_dim=embed_size,
                                      padding_idx=dataset.pad_id)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, dataset.vocab_size)


    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        # logits = torch.randn(
            # indices.shape[0], indices.shape[1], self.vocab_size,
            # device=indices.device
        # )
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        # packed_embeddings = pack_padded_sequence(self.embedding(indices[:, :max(lengths)]), lengths, batch_first=True, enforce_sorted=False)

        rnn_output, _ = self.rnn(self.embedding(indices[:, :max(lengths)]))
        # rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, padding_value=self.dataset.pad_id)

        logits = self.linear(rnn_output)
        return logits


    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """

        # encode prefix
        # tokens = self.dataset.encode(prefix)[:-1]
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = next(self.parameters()).device
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)

        # generate hidden for prefix
        # embeds = self.embedding(tokens)
        embeds = self.embedding(torch.Tensor(tokens).reshape((1, -1)).int().to(device))
        output, hidden = self.rnn(embeds)
        logits = self.linear(output) / temp

        # sample new token from logits
        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        # 2 stopping conditions: reaching max len and getting <eos> token
        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            # process newly obtained token
            embeds = self.embedding(new_tokens)
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output) / temp
            # sample the next token from logits
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        # decode result to a string
        # return self.dataset.decode(tokens.squeeze())
        generated = self.dataset.ids2text(tokens.squeeze())

        return generated

