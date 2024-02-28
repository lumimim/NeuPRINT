import torch
from torch.nn.parameter import Parameter

class time_invariant_permutation_invariant_linear_recon(torch.nn.Module):
    def __init__(self, neuron_dim, time_dim, use_population = False, use_neighbor = False, input_dim = 1, embedding_dim = 32, feature_type = 'embedding'):
        super(time_invariant_permutation_invariant_linear_recon, self).__init__()
        self.neuron_dim  = neuron_dim
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.feature_type = feature_type
        self.use_population = use_population
        self.use_neighbor = use_neighbor
        self.intrisic_mean = Parameter(torch.FloatTensor(1, self.time_dim - 1, (self.embedding_dim + self.input_dim)).uniform_(-1, 1))
        self.intrisic_logvar = Parameter(torch.FloatTensor(1, self.time_dim - 1, (self.embedding_dim + self.input_dim)).uniform_(-1, 1))

    def forward(self, embedding, x, x_pop, x_neighbor, unique_neuron_ids):
        trial_num = x.shape[0]
        x = x.unsqueeze(3) # K trials, N neurons, T time, 1 feature
        if self.use_population:
            x_pop = torch.transpose(x_pop, 1, 2)
            x_pop_repeats = x_pop.unsqueeze(1).repeat(1, self.neuron_dim, 1, 1)
            input = torch.cat((x, x_pop_repeats), dim = -1)
        if self.use_neighbor:
            x_neighbor_reshape = x_neighbor.transpose(2,3)
            input = torch.cat((input, x_neighbor_reshape), dim = -1)
        else:
            input = x
        embedding_repeats = embedding[unique_neuron_ids.long()].unsqueeze(0).unsqueeze(2).repeat(trial_num, 1, self.time_dim, 1) ###
        input = torch.cat((input, embedding_repeats), dim = -1)
        output_mean = torch.mean(torch.mean(self.intrisic_mean.repeat(self.neuron_dim, 1, 1) * input[:,:,:-1,:], dim = -1), dim = -1)
        output_logvar = torch.mean(torch.mean(self.intrisic_logvar.repeat(self.neuron_dim, 1, 1) * input[:,:,:-1,:], dim = -1), dim = -1)
        return output_mean, output_logvar

class time_invariant_permutation_invariant_nonlinear_recon(torch.nn.Module):
    def __init__(self, neuron_dim, time_dim, use_population = False, use_neighbor = False, input_dim = 1, embedding_dim = 32, feature_type = 'embedding'):
        super(time_invariant_permutation_invariant_nonlinear_recon, self).__init__()
        self.neuron_dim  = neuron_dim
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.feature_type = feature_type
        self.use_population = use_population
        self.use_neighbor = use_neighbor
        self.relu = torch.nn.ReLU()
        self.intrisic_mean = Parameter(torch.FloatTensor(1, self.time_dim - 1, (self.embedding_dim + self.input_dim)).uniform_(-1, 1))
        self.intrisic_logvar = Parameter(torch.FloatTensor(1, self.time_dim - 1, (self.embedding_dim + self.input_dim)).uniform_(-1, 1))

    def forward(self, embedding, x, x_pop, x_neighbor, unique_neuron_ids):
        trial_num = x.shape[0]
        x = x.unsqueeze(3) # K trials, N neurons, T time, 1 feature
        if self.use_population:
            x_pop = torch.transpose(x_pop, 1, 2)
            x_pop_repeats = x_pop.unsqueeze(1).repeat(1, self.neuron_dim, 1, 1)
            input = torch.cat((x, x_pop_repeats), dim = -1)
        if self.use_neighbor:
            x_neighbor_reshape = x_neighbor.transpose(2,3)
            input = torch.cat((input, x_neighbor_reshape), dim = -1)
        else:
            input = x
        embedding_repeats = embedding[unique_neuron_ids.long()].unsqueeze(0).unsqueeze(2).repeat(trial_num, 1, self.time_dim, 1) ###
        input = torch.cat((input, embedding_repeats), dim = -1)
        output_mean = torch.mean(torch.mean(self.relu(self.intrisic_mean.repeat(self.neuron_dim, 1, 1) * input[:,:,:-1,:]), dim = -1), dim = -1)
        output_logvar = torch.mean(torch.mean(self.relu(self.intrisic_logvar.repeat(self.neuron_dim, 1, 1) * input[:,:,:-1,:]), dim = -1), dim = -1)

        return output_mean, output_logvar

class time_invariant_permutation_invariant_rnn_recon(torch.nn.Module):
    def __init__(self, neuron_dim, time_dim, use_population = False, use_neighbor = False, input_dim = 1, hidden_dim = 32, embedding_dim = 32, layer_dim = 2):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%self.device)
        super(time_invariant_permutation_invariant_rnn_recon, self).__init__()
        self.neuron_dim = neuron_dim
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.layer_dim = layer_dim
        self.embedding_dim = embedding_dim
        self.use_population = use_population
        self.use_neighbor = use_neighbor
        self.gru = torch.nn.GRU(
            input_size = (self.input_dim + self.embedding_dim), 
            hidden_size = self.hidden_dim, 
            num_layers = self.layer_dim, 
            batch_first = True)
        self.fc_mean = torch.nn.Linear(self.hidden_dim, 1)
        self.fc_logvar = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, embedding, x, x_pop, x_neighbor, unique_neuron_ids):
        trial_num = x.shape[0]
        x = x.reshape((trial_num * self.neuron_dim, self.time_dim, 1))
        batch_size = x.shape[0]
        
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(self.device)
        embedding_repeats = embedding[unique_neuron_ids.long()].unsqueeze(1).repeat(trial_num, self.time_dim, 1) ###
        input_x = torch.zeros(trial_num * self.neuron_dim, self.time_dim, 1).to(self.device)
        input_x[:,1:,:] = x[:,:-1,]
        input = torch.cat((input_x, embedding_repeats), dim = 2)
        if self.use_population:
            x_pop = torch.transpose(x_pop, 1, 2)
            x_pop_repeats = x_pop.unsqueeze(1).repeat(1, self.neuron_dim, 1, 1)
            population_feature_dim = x_pop_repeats.shape[-1]
            x_pop_repeats = x_pop_repeats.reshape(trial_num * self.neuron_dim, self.time_dim, population_feature_dim)
            input = torch.cat((input, x_pop_repeats), dim = 2)
        if self.use_neighbor:
            x_neighbor_reshape = x_neighbor.view(-1,*x_neighbor.shape[2:]).transpose(1,2)
            input = torch.cat((input, x_neighbor_reshape), dim = 2)
        output, hn = self.gru(input, h0)
        output_means = self.fc_mean(output).reshape((trial_num, self.neuron_dim, self.time_dim))
        output_logvars = self.fc_logvar(output).reshape((trial_num, self.neuron_dim, self.time_dim))
        return output_means, output_logvars

import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
          pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:-1]
        else:
          pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class time_invariant_permutation_invariant_transformer_recon(torch.nn.Module):
    def __init__(self, time_dim, input_dim = 1, hidden_dim = 32, embedding_dim = 32, population_feature_dim=4, layer_dim = 4, nhead=2,
                mask_ratio=0.25, mask_last_step_only=False, use_population=True, full_context=True, full_causal=False, context_backward=-1, context_forward=-1, attend_to_self=True, use_neighbor=False): ###
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%self.device)
      super(time_invariant_permutation_invariant_transformer_recon, self).__init__()
      self.time_dim = time_dim
      self.input_dim  = input_dim
      self.hidden_dim = hidden_dim
      self.embedding_dim = embedding_dim
      self.layer_dim = layer_dim
      self.nhead = nhead
      self.mask_ratio = mask_ratio
      self.mask_last_step_only = mask_last_step_only
      self.use_population = use_population ###
      self.full_context = full_context
      self.full_causal = full_causal
      self.context_backward = context_backward
      self.context_forward = context_forward
      self.attend_to_self = attend_to_self
      self.use_neighbor = use_neighbor ###

      self.input_embedder = torch.nn.Linear(1, self.hidden_dim) # i -> h
      self.d_model = self.input_dim + self.embedding_dim # if concat along feature dimension
      self.pos_encoder = PositionalEncoding(d_model=self.d_model)
      self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, norm_first=True, batch_first=True) # dim_feedforward?
      self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.layer_dim)
      self.fc_mean = torch.nn.Linear(self.d_model, 1)
      self.fc_logvar = torch.nn.Linear(self.d_model, 1)

    def forward(self, embedding, x, x_pop, x_neighbor, unique_neuron_ids):
      # x: b x n x t
      num_trials = x.shape[0] # b ###
      num_neurons = x.shape[1] # n ###
      # mask x here, remember to always mask last step of x if we  want to extract loss on last step
      x, mask = mask_batch(x, mask_ratio=self.mask_ratio, mask_last_step_only=self.mask_last_step_only) # b x n x t ###
      x = x.reshape((num_trials * num_neurons, self.time_dim, 1)) # bn x t x 1 ###
      batch_size = x.shape[0]
      embedding_repeats = embedding[unique_neuron_ids.long()].unsqueeze(1).repeat(num_trials, self.time_dim, 1) # n x e -> n x 1 x e -> bn x t x e # if concat along feature dimension ###
      x = self.input_embedder(x) # bn x t x 1 -> bn x t x h
      x = torch.cat((x, embedding_repeats), dim = -1) # bn x t x (h+e) # if concat along feature dimension
      if self.use_population: ###
        x_pop = torch.transpose(x_pop, 1, 2) # b x p x t -> b x t x p
        x_pop_repeats = x_pop.unsqueeze(1).repeat(1, num_neurons, 1, 1) # b x t x p -> b x 1 x t x p -> b x n x t x p ###
        population_feature_dim = x_pop_repeats.shape[-1] # p
        x_pop_repeats = x_pop_repeats.reshape(num_trials * num_neurons, self.time_dim, population_feature_dim) # b x n x t x p -> bn x t x p ###
        x = torch.cat((x, x_pop_repeats), dim = -1) # bn x t x (h+e+p) # if concat along feature dimension
      if self.use_neighbor: ###
        # x_neighbor: bsz x neurons x neighbors x window_len
        x_neighbor_reshape = x_neighbor.view(-1,*x_neighbor.shape[2:]).transpose(1,2) # bsz*neurons x window_len x neighbors
        x = torch.cat((x, x_neighbor_reshape), dim = 2) ###
      x = x * math.sqrt(x.shape[-1]) # bn x t x (*) -> bn x t x (*) 
      x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
      context_mask = generate_context_mask(x, full_context=self.full_context, full_causal=self.full_causal, context_backward=self.context_backward, context_forward=self.context_forward, attend_to_self=self.attend_to_self) ###T
      output = self.transformer_encoder(x, mask=context_mask) # bn x (t+1) x h or bn x t x (h+e+p)
      output_means = self.fc_mean(output).reshape((num_trials, num_neurons, self.time_dim)) # bn x t x (h+e+p) -> bn x t x 1 -> b x n x t # if concat along feature dimension ###
      output_logvars = self.fc_logvar(output).reshape((num_trials, num_neurons, self.time_dim)) # bn x t x (h+e+p) -> bn x t x 1 -> b x n x t # if concat along feature dimension ###
      return output_means, output_logvars, mask

def mask_batch(batch, mask_ratio=0.25, mask_token_ratio=0.8, mask_random_ratio=0.5, mask_last_step_only=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = batch.clone() # make sure we don't corrupt the input data (which is stored in memory) # b x n x t
    if mask_last_step_only:
      mask_ratio = 0.0
    mask_probs = torch.full(batch.shape, mask_ratio, device=device) # 'full' mask conveniently works for us since neurons are considered individually, each neuron will have random timesteps masked independent of other neurons
    mask_probs[:,-1:,:] = torch.ones((mask_probs.shape[0], 1, mask_probs.shape[-1]), device=device) # always mask the last time step
    # If we want any tokens to not get masked, do it here (but we don't currently have any)
    mask = torch.bernoulli(mask_probs)
    mask = mask.bool()
    # We use random assignment so the model learns embeddings for non-mask tokens, and must rely on context
    # Most times, we replace tokens with MASK token
    indices_replaced = torch.bernoulli(torch.full(batch.shape, mask_token_ratio, device=device)).bool() & mask
    batch[indices_replaced] = 0
    # Random % of the time, we replace masked input tokens with random value (the rest are left intact)
    indices_random = torch.bernoulli(torch.full(batch.shape, mask_random_ratio, device=device)).bool() & mask & ~indices_replaced
    random_activity = torch.randn(batch.shape, dtype=torch.float, device=device)
    batch[indices_random] = random_activity[indices_random]
    # Leave the other 10% alone
    mask = torch.where(mask==True, 1.0, 0.0)
    return batch, mask

def generate_context_mask(src, full_context=True, full_causal=False, context_backward=-1, context_forward=-1, attend_to_self=True): ###
    size = src.shape[1] # T
    if full_context:
        # mask = None
        mask = torch.ones(size, size, dtype=torch.bool, device=src.device)
    elif full_causal:
        mask = torch.triu(torch.ones(size, size, device=src.device), diagonal=1) == 0
    else:
        mask = torch.eye(size, device=src.device) == 1
        if context_forward >= 0:
            fwd_mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_forward) == 1).transpose(0, 1)
        else:
            fwd_mask = torch.ones(size, size, dtype=torch.bool, device=src.device)
        if context_backward >= 0:
            bwd_mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_backward) == 1)
        else:
            bwd_mask = torch.ones(size, size, dtype=torch.bool, device=src.device)
        mask = fwd_mask & bwd_mask
    if not attend_to_self:
        mask = mask & (torch.eye(size, device=src.device) == 0)
        if full_causal:
            mask[0,0] = True # so that first time step has someone to attend to ### 
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask