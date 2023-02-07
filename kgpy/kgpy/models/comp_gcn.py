"""
Implementation of CompGCN

See paper for more details - https://arxiv.org/abs/1911.03082
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

from .base.base_gnn_model import BaseGNNModel



class mlp_layer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, device='cuda', bias=True):
        super().__init__()

        self.act = torch.tanh
        self.use_bias = bias

        self.W_entity	= get_param((in_channels, out_channels), device)
        self.W_relation = get_param((in_channels, out_channels), device)
        self.bn	= torch.nn.BatchNorm1d(out_channels)

        if self.use_bias:
            self.register_parameter('mlp_bias', nn.Parameter(torch.zeros(out_channels).to(device)))


    def forward(self, x, r):
        out =  torch.mm(x, self.W_entity)

        if self.use_bias: 
            out = out + self.mlp_bias
        out = self.bn(out)

        if self.act is not None:
            out = self.act(out)

        return out, torch.matmul(r, self.W_relation)	



class CompGCN(BaseGNNModel):
    """
    CompGCN implementation
    """
    def __init__(
        self, 
        num_entities, 
        num_relations, 
        edge_index, 
        edge_type,
        comp_func="corr",
        decoder="conve",
        num_bases=0,
        num_layers=2,
        gcn_dim=200,
        emb_dim=200, 
        weight_init="normal",
        loss_fn="bce",
        device='cuda',
        layer1_drop=.3,
        layer2_drop=.3,
        gcn_drop=.1,

        mlp=False,

        # Only applicable when decoder = 'transe'
        margin=9,   

        # Only applicable for conve...
        num_filters=200,
        conve_drop1=.3,
        conve_drop2=.3,
        ker_size=7,    # kernel size
        ker_height=20, # kernel height
        ker_width=10,   # kernel width
        **kwargs
    ):
        super().__init__(
            type(self).__name__,
            edge_index, 
            edge_type,
            num_layers,
            gcn_dim,
            num_entities, 
            num_relations, 
            emb_dim, 
            weight_init, 
            loss_fn,
            device
        )
        self.margin = margin
        self.num_layers = num_layers
        self.decoder = decoder.lower()
        self.comp_func = comp_func.lower()
        self.gcn_drop = torch.nn.Dropout(gcn_drop)
        self.act = torch.tanh

        self.mlp_agg = mlp

        if num_bases > 0:
            raise NotImplementedError("TODO: Basis convolution")
        else:
            self.conv1 = CompGCNConv(self.emb_dim, self.gcn_dim,  num_relations, self.comp_func, act=self.act, dropout=layer1_drop, device=self.device)

        if self.num_layers == 2:
            self.conv2 = CompGCNConv(self.gcn_dim, self.emb_dim, num_relations, self.comp_func, act=self.act, dropout=layer2_drop, device=self.device)

        self.register_parameter('bias', nn.Parameter(torch.zeros(self.num_entities).to(self.device)))

        # Additional params for conve
        if self.decoder == "conve":
            self.conve_drop1 = torch.nn.Dropout(conve_drop1)
            self.conve_drop2 = torch.nn.Dropout(conve_drop2)

            self.bn0 = torch.nn.BatchNorm2d(1)
            self.bn1 = torch.nn.BatchNorm2d(num_filters)
            self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
            
            self.conve_conv = torch.nn.Conv2d(1, out_channels=num_filters, kernel_size=(ker_size, ker_size), stride=1, padding=0) #, bias=self.p.bias)

            self.ker_height, self.ker_width = ker_height, ker_width
            flat_sz_h     = int(2*ker_width) - ker_size + 1
            flat_sz_w     = ker_height - ker_size + 1
            self.flat_sz  = flat_sz_h * flat_sz_w * num_filters
            self.conve_fc = torch.nn.Linear(self.flat_sz, self.emb_dim)


    def forward(self, triplets, **kwargs):
        """
        Override of prev implementation.

        Only performs 1-N training *with* inverse relation

        Parameters:
        -----------
            triplets: list
                List of triplets to train on

        Returns:
        --------
        Tensor
            preds for batch
        """
        # NOTE: Idk. This is how they have it implemented
        if self.decoder != 'transe':
            r = self.rel_embs 
        else:
            r = torch.cat([self.rel_embs, -self.rel_embs], dim=0)
		
        if self.mlp_agg:
            x, r = self.mlp1(self.ent_embs, r)
        else:
            x, r = self.conv1(self.ent_embs, self.edge_index, self.edge_type, rel_embed=r)

        x = self.gcn_drop(x)
    
        if self.num_layers > 1:
            if self.mlp_agg:
                x, r = self.mlp2(x, r)
            else:
                x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)

            x = self.gcn_drop(x) 	
        
        return self.get_preds(triplets, x, r)


    def get_preds(self, triplets, ent_embs, rel_embs):
        """
        Get predictions of batch against all possible entities
        """
        sub_emb	= torch.index_select(ent_embs, 0, triplets[:, 1])
        rel_emb	= torch.index_select(rel_embs, 0, triplets[:, 0])

        if self.decoder == "transe":
            obj_emb	= sub_emb + rel_emb
            out	= self.margin - torch.norm(obj_emb.unsqueeze(1) - ent_embs, p=1, dim=2)

        elif self.decoder == "distmult":
            obj_emb	= sub_emb * rel_emb
            out = torch.mm(obj_emb, ent_embs.transpose(1, 0))
            out += self.bias.expand_as(out) 

        elif self.decoder == "conve":
            stk_inp	= self.concat(sub_emb, rel_emb)
            out = self.bn0(stk_inp)
            out = self.conve_conv(out)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.conve_drop1(out)
            out = out.view(-1, self.flat_sz)
            out = self.conve_fc(out)
            out = self.conve_drop2(out)
            out = self.bn2(out)
            out = F.relu(out)

            out = torch.mm(out, ent_embs.transpose(1,0))
            out += self.bias.expand_as(out)
        else:
            raise ValueError(f"Invalid decoder for CompGCN - `{self.decoder}`!")
        
        # No need to pass through sigmoid since loss (bce_w_logits) applies it
        return out
    

    def concat(self, e1_embed, rel_embed):
        """
        Only used when decoder = "conve" to convert the embeddings into an "image"
        """
        e1_embed  = e1_embed. view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.ker_width, self.ker_height))

        return stack_inp




#######################################################################################
#
# Convolution layers with and without basis for relations
#
#######################################################################################


class CompGCNConv(MessagePassing):
    """
    CompGCN Convolution.

    Via - https://github.com/malllabiisc/CompGCN/blob/master/model/compgcn_conv.py
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_rels, 
        comp_func,
        dropout=.1,
        act=lambda x:x, 
        bias=False,
        device='cuda'
    ):
        super(self.__class__, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device	= device
        self.bias = bias
        self.comp_func = comp_func

        self.w_loop	  = get_param((in_channels, out_channels), self.device)
        self.w_in	  = get_param((in_channels, out_channels), self.device)
        self.w_out	  = get_param((in_channels, out_channels), self.device)
        self.w_rel 	  = get_param((in_channels, out_channels), self.device)
        self.loop_rel = get_param((1, in_channels), self.device)

        self.drop = torch.nn.Dropout(dropout)
        self.bn	= torch.nn.BatchNorm1d(out_channels)

        if bias: 
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels).to(self.device)))


    def forward(self, x, edge_index, edge_type, rel_embed): 
        """
        Aggregate for the batch
        """
        num_ent   = x.size(0)
        num_edges = edge_index.size(1) // 2
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type,  out_type  = edge_type[:num_edges], edge_type [num_edges:]

        loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        loop_type  = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

        in_norm  = self.compute_norm(in_index,  num_ent)
        out_norm = self.compute_norm(out_index, num_ent)

        in_res	 = self.propagate(in_index,  x=x, edge_type=in_type, rel_embed=rel_embed, edge_norm=in_norm, mode='in')
        loop_res = self.propagate(loop_index, x=x, edge_type=loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res	 = self.propagate(out_index,  x=x, edge_type=out_type,  rel_embed=rel_embed, edge_norm=out_norm, mode='out')

        out	= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

        if self.bias: 
            out = out + self.bias
        out = self.bn(out)
    
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted



    def rel_transform(self, ent_embed, rel_embed):
        """
        Compositional function used in aggregation
        """
        if self.comp_func == 'corr': 	
            trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.comp_func == 'sub': 
            trans_embed  = ent_embed - rel_embed
        elif self.comp_func == 'mult': 	
            trans_embed  = ent_embed * rel_embed
        else: 
            raise NotImplementedError

        return trans_embed


    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        """
        Message that we are aggregating
        """
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)

        out	= torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)



    def update(self, aggr_out):
        return aggr_out


    def compute_norm(self, edge_index, num_ent):
        """
        Compute the normalized Adj matrix
        """
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg	= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv	= deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm




# TODO: Likely not needed anyways since the best performance is without bases

# class CompGCNConvBasis(MessagePassing):
#     """
#     Convolution with basis.

#     Via - https://github.com/malllabiisc/CompGCN/blob/master/model/compgcn_conv_basis.py
#     """
#     def __init__(self, in_channels, out_channels, num_rels, num_bases, act=lambda x:x, cache=True, params=None):
#         super(self.__class__, self).__init__()

#         self.p 			= params
#         self.in_channels	= in_channels
#         self.out_channels	= out_channels
#         self.num_rels 		= num_rels
#         self.num_bases 		= num_bases
#         self.act 		= act
#         self.device		= None
#         self.cache 		= cache			# Should be False for graph classification tasks

#         self.w_loop		= get_param((in_channels, out_channels));
#         self.w_in		= get_param((in_channels, out_channels));
#         self.w_out		= get_param((in_channels, out_channels));

#         self.rel_basis 		= get_param((self.num_bases, in_channels))
#         self.rel_wt 		= get_param((self.num_rels*2, self.num_bases))
#         self.w_rel 		= get_param((in_channels, out_channels))
#         self.loop_rel 		= get_param((1, in_channels));

#         self.drop		= torch.nn.Dropout(self.p.dropout)
#         self.bn			= torch.nn.BatchNorm1d(out_channels)
        
#         self.in_norm, self.out_norm,
#         self.in_index, self.out_index,
#         self.in_type, self.out_type,
#         self.loop_index, self.loop_type = None, None, None, None, None, None, None, None

#         if self.p.bias: self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))

#     def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
#         if self.device is None:
#             self.device = edge_index.device

#         rel_embed = torch.mm(self.rel_wt, self.rel_basis)
#         rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

#         num_edges = edge_index.size(1) // 2
#         num_ent   = x.size(0)

#         if not self.cache or self.in_norm == None:
#             self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
#             self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

#             self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
#             self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

#             self.in_norm     = self.compute_norm(self.in_index,  num_ent)
#             self.out_norm    = self.compute_norm(self.out_index, num_ent)
        
#         in_res	= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
#         loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
#         out_res	= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
#         out	= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

#         if self.p.bias: 
#             out = out + self.bias
#         if self.b_norm: 
#             out = self.bn(out)

#         return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]


#     def rel_transform(self, ent_embed, rel_embed):
#         if self.p.opn == 'corr': 	
#             trans_embed  = ccorr(ent_embed, rel_embed)
#         elif self.p.opn == 'sub': 
#             trans_embed  = ent_embed - rel_embed
#         elif self.p.opn == 'mult': 	
#             trans_embed  = ent_embed * rel_embed
#         else: 
#             raise NotImplementedError

#         return trans_embed


#     def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
#         weight 	= getattr(self, 'w_{}'.format(mode))
#         rel_emb = torch.index_select(rel_embed, 0, edge_type)
#         xj_rel  = self.rel_transform(x_j, rel_emb)
#         out	= torch.mm(xj_rel, weight)

#         return out if edge_norm is None else out * edge_norm.view(-1, 1)


#     def update(self, aggr_out):
#         return aggr_out


#     def compute_norm(self, edge_index, num_ent):
#         row, col	= edge_index
#         edge_weight 	= torch.ones_like(row).float()
#         deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
#         deg_inv		= deg.pow(-0.5)							# D^{-0.5}
#         deg_inv[deg_inv	== float('inf')] = 0
#         norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

#         return norm




#######################################################################################
#
# Helper functions use for compgcn. 
# Via - https://github.com/malllabiisc/CompGCN/blob/master/helper.py
#
#######################################################################################


def get_param(shape, device):
	param = nn.Parameter(torch.Tensor(*shape).to(device))
	nn.init.xavier_normal_(param.data)
	return param

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

# NOTE: Deprecated
# def cconv(a, b):
# 	return torch.fft.irfft(com_mult(torch.fft.rfft(a, 1), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

# def ccorr(a, b):
# 	return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a, 1)), torch.fft.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def cconv(a, b):
	return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b, 1), n=a.shape[-1])

def ccorr(a, b):
    return torch.fft.irfft(conj(torch.fft.rfft(a)) * torch.fft.rfft(b, 1), n=a.shape[-1])
