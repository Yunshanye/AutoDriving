from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSETest, maskedNLLTest, generate_trajectory_heatmap, generate_time_shifted_prediction
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## 网络参数配置
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13, 3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False
metric = 'rmse'

# 
net = highwayNet(args)
net.load_state_dict(torch.load('models/model2.tar'))
if args['use_cuda']:
    net = net.cuda()

# 
tsSet = ngsimDataset('data/TestSet1.mat')
tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)

# 
lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()


# 在测试集上评估模型
for i, data in enumerate(tsDataloader):
    start_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()
    if metric == 'nll':
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask, use_maneuvers=False)
    else:
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedMSETest(fut_pred, fut, op_mask)

    # 累加损失值
    lossVals += l.detach()
    counts += c.detach()
if metric == 'nll':
    print(lossVals / counts)
else:
    # 计算RMSE并转换feet为meter
    print(torch.pow(lossVals / counts, 0.5) * 0.3048)




with torch.no_grad():
    if args['use_maneuvers']:
        fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
        print("fut_pred是一个列表，包含{}个元素".format(len(fut_pred)))
        print("每个元素的尺寸:", fut_pred[0].shape)
        
        # 保存到Excel (保存第一个批次的数据作为示例)
        sample_idx = 0  # 选择批次中的第一个样本
        
        all_data = []
        
        # 计算时间步长 (0.2秒一步)
        time_steps = [t * 0.2 for t in range(fut_pred[0].shape[0])]
        
        # 遍历每个机动组合
        for i, pred in enumerate(fut_pred):
            lat_idx = i % 3  # 横向机动索引 (0,1,2)
            lon_idx = i // 3  # 纵向机动索引 (0,1)
            
            lat_maneuver = ["Keep Lane", "Left Lane Change", "Right Lane Change",][lat_idx]
            lon_maneuver = ["Normal Driving", "Braking"][lon_idx]
            
            # 提取该样本在该机动组合下的预测
            pred_sample = pred[:, sample_idx, :].cpu().numpy()
            
            # 为每个时间步创建一行数据
            for t in range(pred_sample.shape[0]):
                row = {
                    'Time Step': t,
                    'Time (s)': time_steps[t],
                    'Lateral Maneuver': lat_maneuver,
                    'Longitudinal Maneuver': lon_maneuver,
                    'μX (横向位置)': pred_sample[t, 0],  
                    'μY (纵向位置)': pred_sample[t, 1],  
                    'σX': np.exp(pred_sample[t, 2]),  # 横向不确定性
                    'σY': np.exp(pred_sample[t, 3]),  # 纵向不确定性
                    'ρ': np.tanh(pred_sample[t, 4])   # 相关系数
                }
                all_data.append(row)
        
        # 创建DataFrame并保存到Excel
        df = pd.DataFrame(all_data)
        excel_file = 'trajectory_predictions.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"数据已保存到 {excel_file}")
        

        sample_lat_pred = lat_pred[sample_idx].cpu().numpy()
        sample_lon_pred = lon_pred[sample_idx].cpu().numpy()
        
        print("\n机动概率:")
        print("横向机动: 保持车道={:.2f}, 左变道={:.2f}, 右变道={:.2f}".format(
            sample_lat_pred[0], sample_lat_pred[1], sample_lat_pred[2]))
        print("纵向机动: 正常行驶={:.2f}, 制动={:.2f}".format(
            sample_lon_pred[0], sample_lon_pred[1]))
        
        # 根据真实机动和预测机动计算准确率
        true_lat_idx = torch.argmax(lat_enc[sample_idx]).item()
        true_lon_idx = torch.argmax(lon_enc[sample_idx]).item()
        pred_lat_idx = torch.argmax(lat_pred[sample_idx]).item()
        pred_lon_idx = torch.argmax(lon_pred[sample_idx]).item()
        
        true_lat_man = ["Keep Lane", "Left Lane Change", "Right Lane Change"][true_lat_idx]
        true_lon_man = ["Normal Driving", "Braking"][true_lon_idx]
        pred_lat_man = ["Keep Lane", "Left Lane Change", "Right Lane Change"][pred_lat_idx]
        pred_lon_man = ["Normal Driving", "Braking"][pred_lon_idx]
        
        print("\n真实机动: {} - {}".format(true_lat_man, true_lon_man))
        print("预测机动: {} - {}".format(pred_lat_man, pred_lon_man))
        
    else:
        fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
     
        print("fut_pred的尺寸:", fut_pred.shape)
        
        # 保存到Excel (保存第一个batch的第一个样本)
        sample_idx = 0  # 选择批次中的第一个样本
        pred_sample = fut_pred[:, sample_idx, :].cpu().numpy()
        
        # 计算时间步长 (0.2秒一步)
        time_steps = [t * 0.2 for t in range(pred_sample.shape[0])]
        
        # 创建DataFrame并保存
        rows = []
        for t in range(pred_sample.shape[0]):
            row = {
                'Time Step': t,
                'Time (s)': time_steps[t],
                'μX (横向位置)': pred_sample[t, 0],
                'μY (纵向位置)': pred_sample[t, 1],
                'σX': np.exp(pred_sample[t, 2]),
                'σY': np.exp(pred_sample[t, 3]),
                'ρ': np.tanh(pred_sample[t, 4])
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        excel_file = 'single_prediction.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"数据已保存到 {excel_file}")

generate_trajectory_heatmap(fut_pred, fut, lat_pred, lon_pred, hist=hist, sample_idx=0)

# 原始t0时刻预测
fut_pred_t0, lat_pred_t0, lon_pred_t0 = net(hist, nbrs, mask, lat_enc, lon_enc)
step_forward = 5  # 1秒对应的时间步
t1_position = fut[step_forward, sample_idx, :].clone()

hist_t1 = torch.zeros_like(hist)
hist_t1[:-step_forward] = hist[step_forward:]
hist_t1[-step_forward:] = fut[:step_forward]
for i in range(hist_t1.size(0)):
    hist_t1[i, sample_idx, 0] -= t1_position[0]
    hist_t1[i, sample_idx, 1] -= t1_position[1]
nbrs_t1 = nbrs.clone()
mask_t1 = mask.clone()
# 重新运行模型，获取t1时刻的预测
fut_pred_t1, lat_pred_t1, lon_pred_t1 = net(hist_t1, nbrs_t1, mask_t1, lat_enc, lon_enc)
# 生成对比图
result = generate_time_shifted_prediction(
    fut_pred_t0, fut, lat_pred_t0, lon_pred_t0,
    fut_pred_t1, lat_pred_t1, lon_pred_t1,
    hist=hist, sample_idx=sample_idx,
    save_path='time_shifted_prediction_extended.png'
)
extended_traj = result['extended_traj']