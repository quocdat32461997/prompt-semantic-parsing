import gc
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from heapq import heappop, heappush
from torch.nn.utils.rnn import pad_sequence
import copy


class BeamSearchNode:
    def __init__(
        self,
        token_id: int,
        prev_node: Optional[
            "BeamSearchNode"
        ] = None,  # in Python 3.7 and later, discard quotes around BeamSearchNode for type-hinting
        score: float = 0.0,
        length: int = 0,
    ):
        self.token_id: int = token_id
        self.prev_node: BeamSearchNode = prev_node
        self.score: float = score  # Assume to be already in logp
        self.length: int = length

        # self.seq: Tensor = (
        #    None  # store mostly recent tensor of sequence ending by the current node
        # )

    def eval(self) -> None:
        """
        Perform the normalization to address the beam-search curse issue.
        """
        self.score = self.score / float(self.length - 1 + 1e-6)


class BeamSearch(torch.nn.Module):
    """
    This class is to perform beam-search and update final outputs internally.
    This work is inspired by https://github.com/jojonki/BeamSearch/blob/master/beam.py
    """

    def __init__(
        self,
        beam_size: int,
        alpha: float,
        reward: float,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_queue_size: int,
        min_dec_steps: int,
        n_best: int,
        max_seq_len: int,
    ) -> None:
        super(BeamSearch, self).__init__()
        self.beam_size: int = beam_size
        self.alpha: float = alpha
        self.reward: float = reward
        self.bos_token_tensor: Tensor = torch.tensor(bos_token_id)
        self.eos_token_id: int = eos_token_id
        self.pad_token_id: int = pad_token_id
        self.max_seq_len: int = max_seq_len

        assert max_queue_size > int(
            n_best * 1.5
        )  # queue_size must be larger than 1.5 of n_best
        self.n_best: int = n_best  # max number of generated sequences
        self.max_queue_size: int = max_queue_size  # max size of the buffer queue
        self.min_dec_steps: int = min_dec_steps  # min number of decodings steps

    def init_decoder_inputs(
        self,
        batch_size: int,
        device: str,
    ) -> None:

        """
        Init and reset search
        """
        if self.bos_token_tensor.device != device:
            self.bos_token_tensor.to(device)

        self.batch_size: int = batch_size

        self.outputs: List[List[BeamSearchNode]] = [[] for _ in range(batch_size)]
        self.buffers: List[List[BeamSearchNode]] = copy.copy(self.outputs)
        self.indices: List[int] = list(range(batch_size))

        # Start the queue
        for bid in range(batch_size):
            # starting node
            node = BeamSearchNode(
                prev_node=None, token_id=self.bos_token_tensor, score=0.0, length=1
            )
            node.eval()  # update score
            heappush(self.buffers[bid], (-node.score, id(node), node))

    def is_search_done(self) -> bool:
        # Stop if meeting EOS or PAD
        # Or when queue is too large
        return (
            True
            if all(
                [len(self.buffers[bid]) > self.max_queue_size for bid in self.indices]
            )
            else False
        )

    def get_decoder_inputs(
        self, step: int, device: str = "cpu"
    ) -> Tuple[Tensor, Tensor]:
        """
        Return:
            decoder_inputs: Tensor of shape [batch_size, seq_len]
            decoder_attn_mask: Tensor of shape [batch_size, seq_len]
            **NOTE***: seq_len is defined by step
        """
        assert step > 0, "Step is assumed to be always larger than 0."
        indices = []

        # Gather non-terminal decoder_inputs
        decoder_input_list = []
        decoder_attn_list = []
        for bid in self.indices:
            if len(self.buffers[bid]) <= self.max_queue_size:
                node: BeamSearchNode = self.buffers[bid][0][-1]  # Get the best node
                decoder_input_list.append(_build_sequence(node))
                decoder_attn_list.append(torch.ones_like(decoder_input_list[-1]))
                indices.append(bid)
        indices = torch.tensor(indices, dtype=torch.long)

        """ Prep decoder inputs """
        # Pad sequences
        decoder_input_list = pad_sequence(decoder_input_list, batch_first=True, padding_value=self.pad_token_id)
        decoder_input_list = torch.concat([decoder_input_list, torch.full((decoder_input_list.shape[0], step - decoder_input_list.shape[-1]), self.pad_token_id)], dim=-1)
        
        # Update decoder_inputs
        decoder_inputs: Tensor = torch.full((self.batch_size, step), self.pad_token_id, 
        )
        decoder_inputs.index_copy_(0, indices, decoder_input_list)
        
        """ Prep decoder attn mask """
        # Pad sequences
        decoder_attn_list: Tensor = pad_sequence(decoder_attn_list, batch_first=True) # padding by zeros by default
        decoder_attn_list = torch.concat([decoder_attn_list, torch.zeros((decoder_attn_list.shape[0], step - decoder_attn_list.shape[-1]))], dim=-1)

        # Udpate decoder_attn_mask
        decoder_attn_mask: Tensor = torch.zeros((self.batch_size, step))
        decoder_attn_mask.index_copy(0, indices, decoder_attn_list.to(decoder_attn_mask.dtype))
        
        return decoder_inputs.to(device), decoder_attn_mask.to(device)

    def get_final_outputs(self) -> Tensor:
        """
        Rebuild the final and best sequences
        Args:
            - device: where to store tensor
        Returns:
            - outputs: Tensor of final output sequences, shape: [batch_size, padded_sequence]
        """
        output_list: List[Tensor] = []

        # Retrieve and build sequences
        for bid, outputs in enumerate(self.outputs):
            # Sort terminal sequences by score
            outputs = sorted(outputs, key=lambda x: x[0], reverse=True)

            # If no sequence ending w/ <EOS>, get from buffers
            node = outputs.pop(0)[-1] if outputs else heappop(self.buffers[bid])[-1]
            # Add <EOS> if not ending with <EOS>
            if node.token_id != self.eos_token_id:
                if node.length < self.max_seq_len:
                    node = BeamSearchNode(
                        token_id=self.eos_token_id,
                        prev_node=node,
                        length=node.length + 1,
                    )
                node.token_id = self.eos_token_id  # sequences always end by <EOS>

            # Rebuild sequence
            output_list.append(_build_sequence(node))

        # Clean-up memory
        del self.buffers
        del self.outputs
        del self.indices
        gc.collect()

        return pad_sequence(output_list, batch_first=True, padding_value=self.pad_token_id)
        
    def forward(self, probs: Tensor) -> None:
        """
        Internally, select the top-k results and update indices only
        Args:
            - probs: Tensor of log-probs. Shape = [batch_size or queue_size, 1, vocab_size]
        """

        # Sort probs
        # Tensors (topk_prob_list & topk_index_list) of shape [batch_size, beam_size]
        topk_prob_list, topk_index_list = torch.topk(probs, self.beam_size)

        for bid in self.indices:
            # Skip beam-search if the queue size is too large
            if len(self.buffers[bid]) > self.max_queue_size:
                continue

            # Get topk results
            topk_probs = topk_prob_list[bid].tolist()
            topk_indices = topk_index_list[bid].tolist()

            # Get previous node
            prev_node = heappop(self.buffers[bid])[-1]

            # Register new topk nodes
            for prob, vocab_id in zip(topk_probs, topk_indices):
                node: BeamSearchNode = BeamSearchNode(
                    prev_node=prev_node,
                    token_id=vocab_id,
                    score=prob + prev_node.score,
                    length=prev_node.length + 1,
                )
                node.eval()  # update score

                # If current ndoe is eos_token_id, then move to the final output
                if node.token_id != self.eos_token_id:
                    heappush(self.buffers[bid], (-node.score, id(node), node))
                elif node.length >= self.min_dec_steps and len(self.outputs[bid]) < self.n_best:
                    self.outputs.append((node.score, node))

def _build_sequence(node: BeamSearchNode, device: str = "cpu"):
    """
    Traverse the beam-search graph to build the sequence
    """
    # Create shallow copy of node
    node: BeamSearchNode = copy.copy(node)

    outputs: List[Tensor] = []

    while node:
        # Get the current node
        outputs.append(
            node.token_id.item()
            if isinstance(node.token_id, torch.Tensor)
            else node.token_id
        )
        node = node.prev_node

    return torch.tensor(outputs[::-1], device=device)
