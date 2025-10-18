import torch
# import torch.nn as nn

class Beamsearch(object):
    """Class for managing internals of beamsearch procedure.

    References:
        General: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        For TSP: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
    """

    def __init__(self, beam_size, batch_size, num_nodes, commodity_list,
                 dtypeFloat=torch.float32, dtypeLong=torch.long, 
                 probs_type='raw'):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            dtypeFloat: Float data type (for GPU/CPU compatibility)
            dtypeLong: Long data type (for GPU/CPU compatibility)
            probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
            commodity_list: commodities list (batch_size, num_commodities, 3)
        """
        super().__init__()

        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.commodity_list = commodity_list
        self.num_commodities = commodity_list.shape[1]
        self.probs_type = probs_type
        
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        
        # Set beamsearch starting nodes
        self.start_nodes_list = commodity_list[:, :, 0] # (batch_size, num_commodities)
        self.target_nodes_list = commodity_list[:, :, 0] # (batch_size, num_commodities)
        self.start_nodes = torch.zeros(batch_size, beam_size, self.num_commodities, dtype=self.dtypeLong)
        self.start_nodes[:, 0, :] = self.target_nodes_list
        self.target_nodes = self.target_nodes_list.unsqueeze(1)  # (batch_size, 1, num_commodities)
        
        # Move tensors to the same device as commodity_list
        device = commodity_list.device
        self.start_nodes = self.start_nodes.to(device)
        self.target_nodes = self.target_nodes.to(device)
        
        # Mask for constructing valid hypothesis
        self.mask = torch.ones(batch_size, beam_size, num_nodes, self.num_commodities, dtype=self.dtypeFloat, device=device)
        self.update_mask(self.start_nodes)  # Mask the starting node of the beam search
        
        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size, self.num_commodities, dtype=self.dtypeFloat, device=device)
        self.all_scores = []
        
        # Backpointers at each time-step
        self.prev_Ks = []
        
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def get_current_state(self):
        """Get the output of the beam at the current timestep.
        """
        current_state = (self.next_nodes[-1].unsqueeze(2)
                     .expand(self.batch_size, self.beam_size, self.num_nodes, self.num_commodities)).to(torch.long)
        return current_state
    def get_current_origin(self):
        """Get the backpointers for the current timestep.
        """
        return self.prev_Ks[-1]

    def advance(self, trans_probs):
        """Advances the beam based on transition probabilities.

        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
        """
        # Compound the previous scores (summing logits == multiplying probabilities)
        if len(self.prev_Ks) > 0:
            if self.probs_type == 'raw':
                beam_lk = trans_probs * self.scores.unsqueeze(2).expand_as(trans_probs)
            elif self.probs_type == 'logits':
                beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            # Only use the starting nodes from the beam
            if self.probs_type == 'raw':
                beam_lk[:, 1:] = torch.zeros_like(beam_lk[:, 1:], dtype=self.dtypeFloat)
            elif self.probs_type == 'logits':
                beam_lk[:, 1:] = -1e20 * torch.ones_like(beam_lk[:, 1:], dtype=self.dtypeFloat)

        # Multiply by mask
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, self.beam_size * self.num_nodes, self.num_commodities) # (batch_size, beam_size * num_nodes, num_commodities)
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, dim=1, largest=True, sorted=True)
        # Update scores
        self.scores = bestScores
        # Update backpointers
        prev_k = bestScoresId // self.num_nodes  # Integer division
        self.prev_Ks.append(prev_k)
        # Update outputs
        new_nodes = bestScoresId % self.num_nodes
        self.next_nodes.append(new_nodes)
        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask = self.mask.gather(1, perm_mask.long())
        # Mask newly added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        """Sets new_nodes to zero in mask.
        """
        # Move mask to the same device as new_nodes
        self.mask = self.mask.to(new_nodes.device)
        arr = torch.arange(self.num_nodes, device=new_nodes.device).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand_as(self.mask)
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        update_mask = 1 - (arr == new_nodes).float()
        self.mask = self.mask * update_mask
        if self.probs_type == 'logits':
            # Convert 0s in mask to inf
            self.mask[self.mask == 0] = 1e20

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, dim=0, descending=True)

    def get_best(self):
        """Get the score and index of the best hypothesis in the beam."""
        scores, ids = self.sort_best()
        return scores[0], ids[0]

    def get_hypothesis(self):
        """Walk back to construct the full hypothesis.

        Args:
            k: Position in the beam to construct (usually 0 for most probable hypothesis)
        """
        assert self.num_nodes == len(self.prev_Ks) + 1

        hyp = -1 * torch.ones(self.batch_size, self.num_nodes, self.num_commodities, dtype=self.dtypeLong)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, self.target_nodes).view(self.batch_size)
            self.target_nodes = self.prev_Ks[j].gather(1, self.target_nodes)
        return hyp
