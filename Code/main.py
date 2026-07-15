from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN import CaMuR
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 20, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=256, help='The dimension of GCN latent layer')
parser.add_argument('--output_dim1', type=int, default=256, help='The dimension of muti latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
parser.add_argument('--net', type=str, default='Lofgof', help='network type')
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--data', type=str, default='mESC', help='data type')


args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
data_type = args.data
net_type = args.net
datasetpath = 'Dataset/Benchmark Dataset/'





def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

# exp_file = '../BL--ExpressionData.csv'
# tf_file = '../TF.csv'
# target_file = '../Target.csv'
#
# train_file = '../Train_set.csv'
# val_file = '../Validation_set.csv'
# test_file='../Test_set.csv'
exp_file = '../'+datasetpath + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/BL--ExpressionData.csv'
text_file = "../"+datasetpath + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/BL--SemData.csv'
promoter_file = "../"+datasetpath + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/BL--DNAPData.csv'
protein_file = "../"+datasetpath + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/BL--proteinData.csv'


tf_file = '../'+datasetpath + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/TF.csv'
target_file = '../'+datasetpath + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/Target.csv'

train_file =  os.getcwd() + '/../' + net_type + '/' + data_type + ' ' + str(args.num) + '/Train_set.csv'
val_file = os.getcwd() + '/../' + net_type + '/' + data_type + ' ' + str(args.num) +  '/Validation_set.csv'
test_file = os.getcwd() + '/../' + net_type + '/' + data_type + ' ' + str(args.num)  + '/Test_set.csv'





tf_embed_path = r'../Result/'+net_type+'/'+data_type+' '+str(args.num)+'/Channel1.csv'
target_embed_path = r'../Result/'+net_type+'/'+data_type+' '+str(args.num)+'/Channel2.csv'
if not os.path.exists('../Result/'+net_type+'/'+data_type+' '+str(args.num)):
    os.makedirs('../Result/'+net_type+'/'+data_type+' '+str(args.num))



data_input = pd.read_csv(exp_file,index_col=0)
text_input = pd.read_csv(text_file,index_col=0).values
promoter_input = pd.read_csv(promoter_file,index_col=0).values
protein_input = pd.read_csv(protein_file,index_col=0).values

loader = load_data(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
text_feature = torch.tensor(text_input).float() 
promoter_feature = torch.tensor(promoter_input).float() 
protein_feature = torch.tensor(protein_input).float()

tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
text_feature = text_feature.to(device)
promoter_feature = promoter_feature.to(device)
protein_feature = protein_feature.to(device)

tf = tf.to(device)


train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf,loop=args.loop)


adj = adj2saprse_tensor(adj)
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
val_data = torch.from_numpy(validation_data)
#
model = CaMuR(input_dim=feature.size()[1],
                input_text_dim=text_feature.size()[1],
                input_promoter_dim=promoter_feature.size()[1],
                input_protein_dim=protein_feature.size()[1],
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                output_dim=args.output_dim,
                output_dim1=args.output_dim1,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction,
                num_nodes=feature.shape[0],
                )


adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
validation_data = val_data.to(device)

# print(model)
optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
# criterion = MultiConTextLoss(num_losses=3).to(device)
early_stopping = EarlyStopping(save_dir='./',patience=5, verbose=True)
model_path = '../model'
if not os.path.exists(model_path):
    os.makedirs(model_path)



for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)


        # train_y = train_y.to(device).view(-1, 1)
        pred1, pred2, pred3 = model(data_feature,text_feature,promoter_feature,protein_feature, adj, train_x)

        #pred = torch.sigmoid(pred)
        if args.flag:
            pred = torch.softmax(pred, dim=1)
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred3 = torch.sigmoid(pred3)
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)
        # pred3 = torch.sigmoid(pred3)

        loss_BCE = F.binary_cross_entropy(pred1, train_y)
        loss_BCE2 = F.binary_cross_entropy(pred2, train_y)
        loss_BCE3 = F.binary_cross_entropy(pred3, train_y)

        loss = loss_BCE + loss_BCE2 + loss_BCE3
        # loss = criterion(loss_BCE, loss_BCE2, loss_BCE3)
        # loss_BCE = loss_BCE + MoE_gate_loss



        # loss_BCE.backward()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # running_loss += loss_BCE.item()
        running_loss += loss.item()


    model.eval()

    score1, score2, score = model(data_feature, text_feature, promoter_feature, protein_feature, adj, validation_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
        score1 = torch.softmax(score1, dim=1)
        score2 = torch.softmax(score2,dim=1)
    else:
        score = torch.sigmoid(score)
        score1 = torch.sigmoid(score1)
        score2 = torch.sigmoid(score2)

    # score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
    early_stopping(AUC, model)

    print('Epoch:{}'.format(epoch + 1),
            'train loss:{}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))
    if early_stopping.early_stop:
        print("Early stopping")
        break;

# torch.save(model.state_dict(), model_path +'/'+net_type+' '+ data_type+' '+str(args.num)+'.pkl')

# model.load_state_dict(torch.load(model_path +'/'+net_type+' '+ data_type+' '+str(args.num)+'.pkl'))

model.eval()
# tf_embed, target_embed = model.get_embedding()
# embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)

score1, score2, score = model(data_feature, text_feature, promoter_feature, protein_feature, adj, test_data)
if args.flag:
    score = torch.softmax(score, dim=1)
    score1 = torch.softmax(score1, dim=1)
    score2 = torch.softmax(score2,dim=1)
else:
    score = torch.sigmoid(score)
    score1 = torch.sigmoid(score1)
    score2 = torch.sigmoid(score2)

# attn_matrix = model.promoter_Translator.layers[0].self_attention._hooked_attn_weights
# attn_numpy = attn_matrix.numpy()
# np.save('congrn_topk_attention.npy', attn_numpy)
# score = torch.sigmoid(score)
# numpy_data = embed.detach().numpy()
# #
# # # 将numpy数组转换为DataFrame
# df = pd.DataFrame(numpy_data)
# #
# # # 将DataFrame保存为CSV文件
# df.to_csv('hESC_feature1.csv', index=False, header=False)


AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1],flag=args.flag)

print('AUC:{}'.format(AUC),
     'AUPRC:{}'.format(AUPR))

























