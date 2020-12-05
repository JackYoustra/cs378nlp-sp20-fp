"""Model classes and model utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cuda, load_cached_embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from allennlp.data.fields import TextField
# from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.elmo import Elmo

options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

def _sort_batch_by_length(tensor, sequence_lengths):
    """
    Sorts input sequences by lengths. This is required by Pytorch
    `pack_padded_sequence`. Note: `pack_padded_sequence` has an option to
    sort sequences internally, but we do it by ourselves.

    Args:
        tensor: Input tensor to RNN [batch_size, len, dim].
        sequence_lengths: Lengths of input sequences.

    Returns:
        sorted_tensor: Sorted input tensor ready for RNN [batch_size, len, dim].
        sorted_sequence_lengths: Sorted lengths.
        restoration_indices: Indices to recover the original order.
    """
    # Sort sequence lengths
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    # Sort sequences
    sorted_tensor = tensor.index_select(0, permutation_index)
    # Find indices to recover the original order
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths))).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


class AlignedAttention(nn.Module):
    """
    This module returns attention scores over question sequences. Details can be
    found in these papers:
        - Aligned question embedding (Chen et al. 2017):
             https://arxiv.org/pdf/1704.00051.pdf
        - Context2Query (Seo et al. 2017):
             https://arxiv.org/pdf/1611.01603.pdf

    Args:
        p_dim: Int. Passage vector dimension.

    Inputs:
        p: Passage tensor (float), [batch_size, p_len, p_dim].
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over question sequences, [batch_size, p_len, q_len].
    """
    def __init__(self, p_dim):
        super().__init__()
        self.linear = nn.Linear(p_dim, p_dim)
        self.relu = nn.ReLU()

    def forward(self, p, q, q_mask):
        # Compute scores
        p_key = self.relu(self.linear(p))  # [batch_size, p_len, p_dim]
        q_key = self.relu(self.linear(q))  # [batch_size, q_len, p_dim]
        scores = p_key.bmm(q_key.transpose(2, 1))  # [batch_size, p_len, q_len]
        # Stack question mask p_len times
        q_mask = q_mask.unsqueeze(1).repeat(1, scores.size(1), 1)
        # Assign -inf to pad tokens
        scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along question length
        return F.softmax(scores, 2)  # [batch_size, p_len, q_len]


class SpanAttention(nn.Module):
    """
    This module returns attention scores over sequence length.

    Args:
        q_dim: Int. Passage vector dimension.

    Inputs:
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over sequence length, [batch_size, len].
    """
    def __init__(self, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, 1)

    def forward(self, q, q_mask):
        # Compute scores
        q_scores = self.linear(q).squeeze(2)  # [batch_size, len]
        # Assign -inf to pad tokens
        q_scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along sequence length
        return F.softmax(q_scores, 1)  # [batch_size, len]


class BilinearOutput(nn.Module):
    """
    This module returns logits over the input sequence.

    Args:
        p_dim: Int. Passage hidden dimension.
        q_dim: Int. Question hidden dimension.

    Inputs:
        p: Passage hidden tensor (float), [batch_size, p_len, p_dim].
        q: Question vector tensor (float), [batch_size, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Logits over the input sequence, [batch_size, p_len].
    """
    def __init__(self, p_dim, q_dim):
        super().__init__()
        self.linear = nn.Linear(q_dim, p_dim)

    def forward(self, p, q, p_mask):
        # Compute bilinear scores
        q_key = self.linear(q).unsqueeze(2)  # [batch_size, p_dim, 1]
        p_scores = torch.bmm(p, q_key).squeeze(2)  # [batch_size, p_len]
        # Assign -inf to pad tokens
        p_scores.data.masked_fill_(p_mask.data, -float('inf'))
        return p_scores  # [batch_size, p_len]


class BaselineReader(nn.Module):
    # use CNN on characters, finetune through BERT (do it on top of BERT) - won't work b/c can't use pretrained embeddings
    # figure out spacy - track 1
    # use CNN after embedding layer, average the embedding layer w/ CNN (input = characters)
    # input to context2query = linear layer of embedding layer & CNN

    """
    Baseline QA Model
    [Architecture]
        0) Inputs: passages and questions
        1) Embedding Layer: converts words to vectors
        2) Context2Query: computes weighted sum of question embeddings for
               each position in passage.
        3) Passage Encoder: LSTM or GRU.
        4) Question Encoder: LSTM or GRU.
        5) Question Attentive Sum: computes weighted sum of question hidden.
        6) Start Position Pointer: computes scores (logits) over passage
               conditioned on the question vector.
        7) End Position Pointer: computes scores (logits) over passage
               conditioned on the question vector.

    Args:
        args: `argparse` object.

    Inputs:
        batch: a dictionary containing batched tensors.
            {
                'passages': LongTensor [batch_size, p_len],
                'questions': LongTensor [batch_size, q_len],
                'start_positions': Not used in `forward`,
                'end_positions': Not used in `forward`,
            }

    Returns:
        Logits for start positions and logits for end positions.
        Tuple: ([batch_size, p_len], [batch_size, p_len])
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        # self.pad_token_id = args.pad_token_id
        self.pad_token_id = 0

        # Initialize embedding layer (1)
        self.elmo = Elmo(options_file, weight_file, 1)
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)

        # Initialize Context2Query (2)
        self.aligned_att = AlignedAttention(256)

        rnn_cell = nn.LSTM if args.rnn_cell_type == 'lstm' else nn.GRU

        # Initialize passage encoder (3)
        self.passage_rnn = rnn_cell(
            512,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        # Initialize question encoder (4)
        self.question_rnn = rnn_cell(
            256,
            args.hidden_dim,
            bidirectional=args.bidirectional,
            batch_first=True,
        )

        self.dropout = nn.Dropout(self.args.dropout)

        # Adjust hidden dimension if bidirectional RNNs are used
        _hidden_dim = (
            args.hidden_dim * 2 if args.bidirectional
            else args.hidden_dim
        )

        # Initialize attention layer for question attentive sum (5)
        self.question_att = SpanAttention(_hidden_dim)

        # Initialize bilinear layer for start positions (6)
        self.start_output = BilinearOutput(_hidden_dim, _hidden_dim)

        # Initialize bilinear layer for end positions (7)
        self.end_output = BilinearOutput(_hidden_dim, _hidden_dim)

    def load_pretrained_embeddings(self, vocabulary, path):
        """
        Loads GloVe vectors and initializes the embedding matrix.

        Args:
            vocabulary: `Vocabulary` object.
            path: Embedding path, e.g. "glove/glove.6B.300d.txt".
        """
        embedding_map = load_cached_embeddings(path)

        # Create embedding matrix. By default, embeddings are randomly
        # initialized from Uniform(-0.1, 0.1).
        embeddings = torch.zeros(
            (len(vocabulary), self.args.embedding_dim)
        ).uniform_(-0.1, 0.1)

        # Initialize pre-trained embeddings.
        num_pretrained = 0
        for (i, word) in enumerate(vocabulary.words):
            if word in embedding_map:
                embeddings[i] = torch.tensor(embedding_map[word])
                num_pretrained += 1

        # Place embedding matrix on GPU.
        self.embedding.weight.data = cuda(self.args, embeddings)

        return num_pretrained

    def sorted_rnn(self, sequences, sequence_lengths, rnn):
        """
        Sorts and packs inputs, then feeds them into RNN.

        Args:
            sequences: Input sequences, [batch_size, len, dim].
            sequence_lengths: Lengths for each sequence, [batch_size].
            rnn: Registered LSTM or GRU.

        Returns:
            All hidden states, [batch_size, len, hid].
        """
        # Sort input sequences
        sorted_inputs, sorted_sequence_lengths, restoration_indices = _sort_batch_by_length(
            sequences, sequence_lengths
        )
        # Pack input sequences
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.long().tolist(),
            batch_first=True
        )
        # Run RNN
        packed_sequence_output, _ = rnn(packed_sequence_input, None)
        # Unpack hidden states
        unpacked_sequence_tensor, _ = pad_packed_sequence(
            packed_sequence_output, batch_first=True
        )
        # Restore the original order in the batch and return all hidden states
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

    def forward(self, batch):

        # 1) Embedding Layer: Embed the passage and question.
        passage_output = self.elmo(batch['passages'])
        question_output = self.elmo(batch['questions'])

        passage_embeddings = torch.stack(passage_output["elmo_representations"]).squeeze()
        question_embeddings = torch.stack(question_output["elmo_representations"]).squeeze()

        # Obtain masks and lengths for passage and question.
        passage_mask = passage_output["mask"].squeeze()
        question_mask = question_output["mask"].squeeze()
        passage_lengths = passage_mask.long().sum(-1)  # [batch_size]
        question_lengths = question_mask.long().sum(-1)  # [batch_size]

        # 2) Context2Query: Compute weighted sum of question embeddings for
        #        each passage word and concatenate with passage embeddings.
        aligned_scores = self.aligned_att(
            passage_embeddings, question_embeddings, ~question_mask
        )  # [batch_size, p_len, q_len]
        aligned_embeddings = aligned_scores.bmm(question_embeddings)  # [batch_size, p_len, q_dim]
        passage_embeddings = cuda(
            self.args,
            torch.cat((passage_embeddings, aligned_embeddings), 2),
        )  # [batch_size, p_len, p_dim + q_dim]

        # 3) Passage Encoder
        passage_hidden = self.sorted_rnn(
            passage_embeddings, passage_lengths, self.passage_rnn
        )  # [batch_size, p_len, p_hid]
        passage_hidden = self.dropout(passage_hidden)  # [batch_size, p_len, p_hid]

        # 4) Question Encoder: Encode question embeddings.
        question_hidden = self.sorted_rnn(
            question_embeddings, question_lengths, self.question_rnn
        )  # [batch_size, q_len, q_hid]

        # 5) Question Attentive Sum: Compute weighted sum of question hidden
        #        vectors.
        question_scores = self.question_att(question_hidden, ~question_mask)
        question_vector = question_scores.unsqueeze(1).bmm(question_hidden).squeeze(1)
        question_vector = self.dropout(question_vector)  # [batch_size, q_hid]

        # 6) Start Position Pointer: Compute logits for start positions
        start_logits = self.start_output(
            passage_hidden, question_vector, ~passage_mask
        )  # [batch_size, p_len]

        # 7) End Position Pointer: Compute logits for end positions
        end_logits = self.end_output(
            passage_hidden, question_vector, ~passage_mask
        )  # [batch_size, p_len]

        return start_logits, end_logits  # [batch_size, p_len], [batch_size, p_len]
