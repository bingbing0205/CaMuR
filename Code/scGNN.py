import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
import math
from torch.distributions.normal import Normal
import numpy as np

class CaMuR(nn.Module):
    def __init__(self,input_dim,input_text_dim,input_promoter_dim,input_protein_dim,hidden1_dim,hidden2_dim,output_dim, output_dim1, num_head1,num_head2,
                 alpha,device,type,reduction,num_nodes):
        super(CaMuR, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.num_nodes=num_nodes

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim

        
        self.promoter_Function_Aggregator = MultiLayerTranslator(input_promoter_dim, output_dim1, num_heads=4, num_layers=1)
        self.protein_Function_Aggregator = MultiLayerTranslator(input_protein_dim, output_dim1, num_heads=4, num_layers=1)
        self.text_Function_Aggregator = MultiLayerTranslator(input_text_dim, output_dim1, num_heads=4, num_layers=1)
        
        self.Adprm = Feature_Inter(output_dim, output_dim)
       
        
        self.Cmcr = MoEgate(input_size=output_dim1, output_size=output_dim1, num_experts=output_dim1*2, k=output_dim1)
       
        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,num_nodes,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)

        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,num_nodes,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        self.tf_linear1 = nn.Linear(hidden2_dim,output_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,output_dim)



        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)




    def Ccpm(self,x,adj):
        if self.reduction =='concate':

            x = torch.cat([att(x, adj,1)for att in self.ConvLayer1], dim=1)
            x = F.elu(x)


        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj,1) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)

        else:
            raise TypeError


        out = torch.mean(torch.stack([att(x, adj,2) for att in self.ConvLayer2]),dim=0)
        out=F.elu(out)

        return out


    def decode(self,tf_embed,target_embed):

        if self.type =='dot':

            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)


            return prob

        elif self.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)

            return prob

        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)

            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))


    def forward(self,x,x_text,x_promoter,x_protein,adj,train_sample):

        embed= self.Ccpm(x,adj)
        
        text_feature = self.text_Function_Aggregator(x_text)
        
        promoter_feature = self.promoter_Function_Aggregator(x_promoter)
        protein_feature = self.protein_Function_Aggregator(x_protein)

        text_feature = F.normalize(text_feature, p=2, dim=1)
        promoter_feature = F.normalize(promoter_feature, p=2, dim=1)
        protein_feature = F.normalize(protein_feature, p=2, dim=1)
       
        promoter_feature_con = torch.concat([text_feature, promoter_feature], dim=1)
        
        gate_promoter = self.Cmcr(text_feature, promoter_feature)
        promoter_feature_1 = gate_promoter * promoter_feature_con
        promoter_feature = promoter_feature_con + promoter_feature_1
        

        protein_feature_con = torch.concat([text_feature, protein_feature], dim=1)
        gate_protein = self.Cmcr(text_feature, protein_feature)
        protein_feature_1 = gate_protein * protein_feature_con
       

        protein_feature = protein_feature_con + protein_feature_1


        tf_embed = self.tf_linear1(embed)
        tf_embed = F.elu(tf_embed)
        tf_embed = F.dropout(tf_embed,p=0.01)
        target_embed = self.target_linear1(embed)
        
        target_embed = F.elu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)

        feature_gate = self.Adprm(target_embed)
       

        tf_embed = tf_embed * feature_gate
        target_embed = target_embed * feature_gate


        promoter_feature_fusion = torch.cat([target_embed, promoter_feature], dim=1)
        protein_feature_fusion = torch.cat([tf_embed, protein_feature], dim=1)

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred1 = self.decode(train_tf, train_target)

        train_tf = protein_feature[train_sample[:,0]]
        train_target = promoter_feature[train_sample[:, 1]]

        pred2 = self.decode(train_tf, train_target)

        train_tf = protein_feature_fusion[train_sample[:,0]]
        train_target = promoter_feature_fusion[train_sample[:, 1]]

        pred3 = self.decode(train_tf, train_target)

        return pred1, pred2, pred3

    def get_embedding(self):
        return self.tf_ouput, self.target_output



class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,nums,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.num=nums

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = torch.exp(-torch.square(Wh1 - Wh2.T)/1e-0)

        return e



    def forward(self,x,adj,layer):

        h=torch.matmul(x,self.weight)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
       
        attention = torch.where(adj.to_dense()>0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, training=self.training)
        
        h_pass = torch.matmul(attention, h)
       
        output_data = h_pass

        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)

        output_data = F.normalize(output_data,p=2,dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data
        
class TranslatorLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(TranslatorLayer, self).__init__()
        
        self.self_attention = TopKMultiheadAttention(input_dim, num_heads)
       
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
       
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, feature):
       
        embedding = self.self_attention(feature, feature, feature)[0]
        embedding = self.ffn(embedding)
        embedding  = self.norm(embedding)

        return embedding 


class MultiLayerTranslator(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(MultiLayerTranslator,self).__init__()

        self.BatchNorm1d = nn.BatchNorm1d(input_dim)
        self.embedding = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([
            TranslatorLayer(output_dim, output_dim, num_heads) for _ in range(num_layers)
        ])
    

    def forward(self, feature):
      
        feature = self.BatchNorm1d(feature)
        feature = self.embedding(feature).unsqueeze(0)
        feature_R = feature

        for layer in self.layers:
             feature = layer(feature)
        feature = feature.squeeze(0) + feature_R.squeeze(0)

        return feature

class TopKMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, k_selection=20, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.k_selection = k_selection
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        
 
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scale = 1.0 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if 0 < self.k_selection < seq_len:
           
            top_val, _ = torch.topk(scores, k=self.k_selection, dim=-1)
            threshold = top_val[..., -1].unsqueeze(-1)
            mask_topk = scores < threshold
            scores = scores.masked_fill(mask_topk, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)
        
        return output, attn_weights

class MoEgate(nn.Module):
    def __init__(self, input_size, output_size, num_experts, k, noisy_gating=True):
        super(MoEgate, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.k = k
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=2)
        
        self.w_gate = nn.Parameter(torch.zeros(input_size, output_size), requires_grad=True)
        self.w2_gate = nn.Parameter(torch.zeros(input_size, output_size),requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, output_size), requires_grad=True)
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(output_size)
        

    def noisy_top_k_gating(self, x, x2, noise_epsilon=1e-2):

        
        x_avg = self.avgpool(x.permute(1, 0))
        x_max = self.maxpool(x.permute(1, 0))
        x_fusion = torch.cat([x_avg, x_max], dim=1)
        x2_avg = self.avgpool(x2.permute(1,0))
        x2_max = self.maxpool(x2.permute(1,0))
        x2_fusion = torch.cat([x2_avg, x2_max], dim=1)
        fusion_concat = torch.stack([x_fusion, x2_fusion], dim=0)
        fusion_weight = self.conv(fusion_concat).squeeze(2)
       
        fusion_weight = self.sigmoid(fusion_weight)
        w1, w2 = torch.chunk(fusion_weight, chunks=2, dim=0)
        
        x = self.norm(x * w1)
        x2 = self.norm(x2 * w2)
       
        
        clean_logits_1 = x @ self.w_gate
        clean_logits_2 = x2 @ self.w2_gate
        
#         8*8 8*4
        if self.noisy_gating :
            raw_noise_stddev_1 = x @ self.w_noise
            raw_noise_stddev_2 = x2 @ self.w_noise
#             8*8 8*4
            noise_stddev_1 = ((self.softplus(raw_noise_stddev_1) + noise_epsilon))
            noise_stddev_2 = ((self.softplus(raw_noise_stddev_2)) + noise_epsilon)
            noisy_logits_1 = clean_logits_1 + (torch.randn_like(clean_logits_1) * noise_stddev_2)
           
            logits_1 = noisy_logits_1
            noisy_logits_2 = clean_logits_2 + (torch.randn_like(clean_logits_2) * noise_stddev_1)
           
            logits_2 = noisy_logits_2
        else:
            logits_1 = clean_logits_1
            logits_2 = clean_logits_2
       
        logits = torch.cat([logits_1, logits_2], dim=1)
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.sigmoid(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # print(gates)

        
        return gates

    def forward(self, x,x2, loss_coef=1e-2):
        """
        x: (Batch_Size, Input_Features) -> 纯二维数据
        x的大小 8*8
        """
        
        gates = self.noisy_top_k_gating(x, x2)

        return gates





class Feature_Inter(nn.Module):
    def __init__(self, input_size, output_size):
        super(Feature_Inter, self).__init__() 

        self.input_size = input_size
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(in_channels = input_size, out_channels = output_size, kernel_size=2)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x_avg = self.avgpool(x.permute(1,0))
        x_max = self.maxpool(x.permute(1,0))

        x_fusion = torch.cat([x_avg, x_max], dim=1).unsqueeze(0)

        fusion_weight = self.conv(x_fusion).view(-1, self.input_size)

        w = self.sigmoid(fusion_weight)
        x = self.norm(x * w)

        return x











