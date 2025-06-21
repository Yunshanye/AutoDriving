from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):


    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid



    def __len__(self):
        return len(self.D)



    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))

        # Maneuvers 'lon_enc' = one-hot vector, 'lat_enc = one-hot vector
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1

        return hist,fut,neighbors,lat_enc,lon_enc



    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist



    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut



    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)


        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.byte()


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)


        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch

#________________________________________________________________________________________________________________________________________





## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

## Batchwise NLL loss, uses mask for variable output lengths
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1):
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1):
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                # If we represent likelihood in feet^(-1):
                out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                # If we represent likelihood in m^(-1):
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # If we represent likelihood in feet^(-1):
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # If we represent likelihood in m^(-1):
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs






def generate_probability_comparison(fut_pred, fut, lat_pred, lon_pred, hist=None, sample_idx=0, save_path='trajectory_prob_comparison.png'):
    """
    基于概率生成轨迹对比图，透明度表示机动概率
    在图中标注概率值
    
    """
    true_traj = fut[:, sample_idx, :].detach().cpu().numpy()
    
    if hist is not None:
        hist_traj = hist[:, sample_idx, :].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 12))
    
    # 绘制历史轨迹
    if hist is not None:
        plt.plot(hist_traj[:, 0], hist_traj[:, 1], 'r--o', linewidth=1.5, markersize=4, alpha=0.7, label='Historical Trajectory')
        
        # 标记分界点
        plt.plot(0, 0, 'ko', markersize=6)
        plt.annotate(
            "(0.00, 0.00)", 
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color='black',
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
        plt.plot(0, 0, 'ko', markersize=6, label='Current Position')
    
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'r-*', linewidth=2, label='True Future Trajectory')
    
    maneuver_probs = []
    modes = [
        "Keep Lane - Normal Driving", 
        "Left Lane Change - Normal Driving", 
        "Right Lane Change - Normal Driving",
        "Keep Lane - Braking", 
        "Left Lane Change - Braking",
        "Right Lane Change - Braking"
    ]
    

    lat_prob = lat_pred[sample_idx].detach().cpu().numpy()  
    lon_prob = lon_pred[sample_idx].detach().cpu().numpy() 
    

    for lon_idx in range(2):
        for lat_idx in range(3):
            prob = lat_prob[lat_idx] * lon_prob[lon_idx]
            maneuver_probs.append(prob)
    
    # 透明度基于概率
    line_styles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', '^', 's', 'D', 'x', '+']
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    max_prob = max(maneuver_probs)
    
    for i, pred in enumerate(fut_pred):
        pred_sample = pred[:, sample_idx, :].detach().cpu().numpy()
        
        # 计算透明度
        alpha = 0.1 + 0.9 * (maneuver_probs[i] / max_prob) if max_prob > 0 else 0.1
        
        plt.plot(
            pred_sample[:, 0], pred_sample[:, 1],
            linestyle=line_styles[i], marker=markers[i], color=colors[i],
            linewidth=1.5, markersize=5, alpha=alpha,
            label=f"{modes[i]}"  
        )
        
        end_x, end_y = pred_sample[-1, 0], pred_sample[-1, 1]
        plt.annotate(
            f"p={maneuver_probs[i]:.2f}", 
            xy=(end_x, end_y),
            xytext=(5, 0),  
            textcoords="offset points",
            fontsize=9,
            color=colors[i],
            weight='bold',
            alpha=min(1.0, alpha + 0.3)  
        )
    
    plt.xlim(-20, 20)
    plt.title('Predicted Trajectories vs True Trajectory (Transparency reflects Probability)')
    plt.xlabel('Lateral Position [feet]')
    plt.ylabel('Longitudinal Position [feet]')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"概率轨迹对比图已保存到 {save_path}")


def generate_trajectory_heatmap(fut_pred, fut, lat_pred, lon_pred, hist=None, sample_idx=0, save_path='trajectory_heatmap.png'):
    """
    为每个机动类型生成热力图，显示未来预测分布。
    
    """
  
    true_traj = fut[:, sample_idx, :].detach().cpu().numpy()
    num_timesteps = true_traj.shape[0]
    
 
    if hist is not None:
        hist_traj = hist[:, sample_idx, :].detach().cpu().numpy()
    
    start_timestep = 10
    if num_timesteps <= start_timestep:
        print(f"警告: 轨迹时间步数量({num_timesteps})小于起始时间步({start_timestep})。使用第一个时间步。")
        start_timestep = 0
    
    modes = [
        "Keep Lane - Normal Driving", 
        "Left Lane Change - Normal Driving", 
        "Right Lane Change - Normal Driving",
        "Keep Lane - Braking", 
        "Left Lane Change - Braking",
        "Right Lane Change - Braking"
    ]
      
    maneuver_probs = []
    lat_prob = lat_pred[sample_idx].detach().cpu().numpy()
    lon_prob = lon_pred[sample_idx].detach().cpu().numpy()
    
    for lon_idx in range(2):
        for lat_idx in range(3):
            prob = lat_prob[lat_idx] * lon_prob[lon_idx]
            maneuver_probs.append(prob)
    
    for i, pred in enumerate(fut_pred):
        plt.figure(figsize=(10, 12))
        
        pred_sample = pred[:, sample_idx, :].detach().cpu().numpy()
        
        min_x, max_x = -20, 20  
        
        min_y = min(0, np.min(true_traj[:, 1]) - 10)
        max_y = max(np.max(true_traj[:, 1]) + 10, np.max(pred_sample[:, 1]) + 10)
        
        # 固定选择时间步（2秒、2.4秒、2.8秒、3.2秒）
        #time_indices = [10, 12, 14, 16]  # 相对于预测开始的时间步
        time_indices = [5, 8, 11, 14]
        time_indices = [t for t in time_indices if t < num_timesteps]
        
        # 计算真实概率
        prob = maneuver_probs[i]

        for idx, t in enumerate(time_indices):
            plt.subplot(2, 2, idx+1)
            
            mu_x, mu_y = pred_sample[t, 0], pred_sample[t, 1]
            sig_x, sig_y = np.exp(pred_sample[t, 2]), np.exp(pred_sample[t, 3])
            rho = np.tanh(pred_sample[t, 4])

            x = np.linspace(min_x, max_x, 100)
            y = np.linspace(min_y, max_y, 100)
            X, Y = np.meshgrid(x, y)
            
            try:
                Z = multivariate_normal.pdf(
                    np.dstack([X, Y]), 
                    mean=[mu_x, mu_y], 
                    cov=[[sig_x**2, rho*sig_x*sig_y], 
                         [rho*sig_x*sig_y, sig_y**2]]
                )
                # 归一化
                Z = Z / Z.max()
    
                from scipy.ndimage import gaussian_filter
                Z = gaussian_filter(Z, sigma=1.0)
   
                min_level = 0.05  
                levels = np.linspace(min_level, 1.0, 20)

                # 绘制热力图
                #contour = plt.contourf(X, Y, Z, cmap='Blues', levels=20)
                contour = plt.contourf(X, Y, Z, levels=levels, cmap='Blues')
                #plt.colorbar(contour, label='Probability Density')
                plt.colorbar(contour, label='Normalized Probability Density')
            except:
                print(f"警告: 时间步 {t} 的协方差矩阵可能不是正定的。跳过热力图绘制。")
                plt.plot(mu_x, mu_y, 'b+', markersize=8)
                plt.plot(true_traj[t, 0], true_traj[t, 1], 'r*', markersize=10)
            

            #if hist is not None:
            #    plt.plot(hist_traj[:, 0], hist_traj[:, 1], 'r--o', linewidth=1, markersize=3, alpha=0.5, label='History')
            
            plt.plot(true_traj[:, 0], true_traj[:, 1], 'r-', linewidth=1.5, alpha=0.5, label='True Future')
            
            plt.plot(true_traj[t, 0], true_traj[t, 1], 'r*', markersize=10, label='True at t')
            
            plt.plot(mu_x, mu_y, 'b+', markersize=8, label='Predicted at t')
            
            plt.plot(0, 0, 'ko', markersize=5, label='Current Position')
            plt.title(f'Time Step {t} ({t*0.2:.1f}s)')
            plt.xlabel('X Position (Lateral)')
            plt.ylabel('Y Position (Longitudinal)')

            plt.xlim(min_x, max_x)
            
            if idx == 0:
                plt.legend(loc='upper right', fontsize='small')
        
        plt.suptitle(f'Trajectory Heatmap: {modes[i]} (Probability: {prob:.3f})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        mode_filename = save_path.replace('.png', f'_mode{i}.png')
        plt.savefig(mode_filename, dpi=300)
        plt.close()
    
    plt.figure(figsize=(12, 10))
    
    #if hist is not None:
    #    plt.plot(hist_traj[:, 0], hist_traj[:, 1], 'r--o', linewidth=1.5, markersize=4, alpha=0.7, label='Historical Trajectory')
    
    plt.plot(0, 0, 'ko', markersize=6, label='Current Position')
    
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'r-*', linewidth=2, label='True Future Trajectory')
    
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', '^', 's', 'D', 'x', '+']
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, pred in enumerate(fut_pred):
        pred_sample = pred[:, sample_idx, :].detach().cpu().numpy()
        plt.plot(
            pred_sample[:, 0], pred_sample[:, 1],
            linestyle=line_styles[i], marker=markers[i], color=colors[i],
            linewidth=1.5, markersize=5, alpha=0.7,
            label=f"{modes[i]} (p={maneuver_probs[i]:.2f})"
        )

    plt.xlim(-20, 20)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    
    plt.title('Predicted Trajectories vs True Trajectory (Next 5 Seconds)')
    plt.xlabel('Lateral Position [feet]')
    plt.ylabel('Longitudinal Position [feet]')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    comparison_path = save_path.replace('.png', '_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    print(f"热力图已保存到 {save_path.replace('.png', '_mode*.png')}")
    print(f"轨迹对比图已保存到 {comparison_path}")

    prob_comparison_path = save_path.replace('.png', '_prob_comparison.png')
    generate_probability_comparison(fut_pred, fut, lat_pred, lon_pred, hist, sample_idx, prob_comparison_path)


    continuous_path = save_path.replace('.png', '_continuous.png')
    generate_continuous_distribution(fut_pred, fut, lat_pred, lon_pred, sample_idx, save_path=continuous_path)


    #gmm_path = save_path.replace('.png', '_gmm.png')
    #generate_gmm_visualization(fut_pred, fut, lat_pred, lon_pred, hist, sample_idx, save_path=gmm_path)


    


'''
def generate_all_timesteps_heatmap(fut_pred, fut, hist=None, sample_idx=0, mode_idx=0, save_path='all_timesteps_heatmap.png'):
    """
    为单一模态(默认mode0)生成包含所有时间步预测分布的可视化
    时间步从0到24 
    """

    true_traj = fut[:, sample_idx, :].detach().cpu().numpy()
    num_timesteps = true_traj.shape[0]

    if hist is not None:
        hist_traj = hist[:, sample_idx, :].detach().cpu().numpy()

    pred_sample = fut_pred[mode_idx][:, sample_idx, :].detach().cpu().numpy()

    modes = [
        "Keep Lane - Normal Driving", 
        "Left Lane Change - Normal Driving", 
        "Right Lane Change - Normal Driving",
        "Keep Lane - Braking", 
        "Left Lane Change - Braking",
        "Right Lane Change - Braking"
    ]
    mode_name = modes[mode_idx]

    plt.figure(figsize=(10, 12))

    min_x, max_x = -20, 20  

    min_y = min(0, np.min(true_traj[:, 1]) - 10) 
    max_y = max(np.max(true_traj[:, 1]) + 10, np.max(pred_sample[:, 1]) + 10)
    

    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    X, Y = np.meshgrid(x, y)
    

    for t in range(num_timesteps):
        mu_x, mu_y = pred_sample[t, 0], pred_sample[t, 1]
        sig_x, sig_y = np.exp(pred_sample[t, 2]), np.exp(pred_sample[t, 3])
        rho = np.tanh(pred_sample[t, 4])
        

        if t < 15:
            sig_x = min(sig_x, 5.0)
            sig_y = min(sig_y, 10.0)
        
        try:
            # 计算多变量高斯分布
            Z = multivariate_normal.pdf(
                np.dstack([X, Y]), 
                mean=[mu_x, mu_y], 
                cov=[[sig_x**2, rho*sig_x*sig_y], 
                     [rho*sig_x*sig_y, sig_y**2]]
            )
            

            #alpha = 0.2 + 0.02 * t  
            #alpha = min(alpha, 0.9)  
            alpha = 0.5

            max_z = np.max(Z)
            min_level = max_z * 0.1  
            levels = np.linspace(min_level, max_z, 20)


            contour = plt.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=alpha)

            if t % 5 == 0 or t == num_timesteps - 1:
                plt.text(mu_x, mu_y, f"{t*0.2:.1f}s", 
                       color='black', fontsize=8, 
                       ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
        except:
            print(f"警告: 时间步 {t} 的协方差矩阵可能不是正定的。跳过该时间步的概率分布。")
    

    if hist is not None:
        plt.plot(hist_traj[:, 0], hist_traj[:, 1], 'r--o', linewidth=1, markersize=3, alpha=0.5, label='History')
    

    plt.plot(true_traj[:, 0], true_traj[:, 1], 'r-', linewidth=1.5, alpha=0.5, label='True Future')
    

    plt.plot(0, 0, 'ko', markersize=6, label='Current Position')

    plt.plot(pred_sample[:, 0], pred_sample[:, 1], 'b+', linewidth=1.5, markersize=4, alpha=0.7, label='Predicted Path')
    

    plt.title(f'Combined Time Steps Trajectory Heatmap: {mode_name}', fontsize=16)
    plt.xlabel('X Position (Lateral)')
    plt.ylabel('Y Position (Longitudinal)')
    plt.xlim(min_x, max_x)
    plt.legend(loc='upper right', fontsize='small')
    plt.colorbar(contour, label='Probability Density')
    plt.grid(True, alpha=0.3)
    plt.suptitle(f'All 25 Time Steps (0-5s) Combined', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"所有时间步组合热力图已保存到 {save_path}")
'''


def generate_continuous_distribution(fut_pred, fut, lat_pred, lon_pred, sample_idx=0, save_path='continuous_distribution.png'):
    """
    生成连续的分布表示
    """
    true_traj = fut[:, sample_idx, :].detach().cpu().numpy()
    
    maneuver_probs = []
    modes = [
        "Keep Lane - Normal Driving", 
        "Left Lane Change - Normal Driving", 
        "Right Lane Change - Normal Driving",
        "Keep Lane - Braking", 
        "Left Lane Change - Braking",
        "Right Lane Change - Braking"
    ]
    
    # 从lat_pred和lon_pred
    lat_prob = lat_pred[sample_idx].detach().cpu().numpy()
    lon_prob = lon_pred[sample_idx].detach().cpu().numpy()
    
    for lon_idx in range(2):
        for lat_idx in range(3):
            prob = lat_prob[lat_idx] * lon_prob[lon_idx]
            maneuver_probs.append(prob)
    max_prob_idx = np.argmax(maneuver_probs)
    max_prob = maneuver_probs[max_prob_idx]
    mode_name = modes[max_prob_idx]    
    pred_sample = fut_pred[max_prob_idx][:, sample_idx, :].detach().cpu().numpy()
    plt.figure(figsize=(10, 12))
    min_x, max_x = -20, 20
    min_y = min(min(0, np.min(true_traj[:, 1])), np.min(pred_sample[:, 1])) - 10
    max_y = max(np.max(true_traj[:, 1]), np.max(pred_sample[:, 1])) + 10
    

    num_interp_points = 4  
    num_orig_points = pred_sample.shape[0]
    num_total_points = (num_orig_points - 1) * (num_interp_points + 1) + 1
    
    t_orig = np.arange(num_orig_points)
    t_interp = np.linspace(0, num_orig_points - 1, num_total_points)
    

    from scipy.interpolate import make_interp_spline
    
    spline_x = make_interp_spline(t_orig, pred_sample[:, 0], k=3)
    spline_y = make_interp_spline(t_orig, pred_sample[:, 1], k=3)
    mu_x_interp = spline_x(t_interp)
    mu_y_interp = spline_y(t_interp)
    
    spline_sig_x = make_interp_spline(t_orig, np.log(np.exp(pred_sample[:, 2])), k=3)
    spline_sig_y = make_interp_spline(t_orig, np.log(np.exp(pred_sample[:, 3])), k=3)
    spline_rho = make_interp_spline(t_orig, np.tanh(pred_sample[:, 4]), k=3)
    sig_x_interp = np.exp(spline_sig_x(t_interp))
    sig_y_interp = np.exp(spline_sig_y(t_interp))
    rho_interp = np.tanh(spline_rho(t_interp))  
    

    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    X, Y = np.meshgrid(x, y)

    from scipy.ndimage import gaussian_filter
    Z_total = np.zeros((100, 100))
    
    for i in range(len(t_interp) - 1, -1, -1):
        mu_x, mu_y = mu_x_interp[i], mu_y_interp[i]
        sig_x, sig_y = sig_x_interp[i], sig_y_interp[i]
        rho = rho_interp[i]
        
        t_value = t_interp[i]
        t_idx = int(t_value)
        
        if t_idx < 15:
            sig_x = min(sig_x, 5.0)
            sig_y = min(sig_y, 10.0)
        
        try:
            Z = multivariate_normal.pdf(
                np.dstack([X, Y]), 
                mean=[mu_x, mu_y], 
                cov=[[sig_x**2, rho*sig_x*sig_y], 
                     [rho*sig_x*sig_y, sig_y**2]]
            )
            
            # 归一化Z
            Z = Z / Z.max()
            
            Z = gaussian_filter(Z, sigma=1.0)
            
            alpha = 0.5
            
            Z_total = np.maximum(Z_total, Z * alpha)
            
        except:
            print(f"警告: 时间步 {t_idx} 的协方差矩阵可能不是正定的。跳过该时间步。")
    
    levels = np.linspace(0.001, 1.0, 20)  
    contour = plt.contourf(X, Y, Z_total, levels=levels, cmap='Blues')
    plt.colorbar(contour, label='Probability Density')
    
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'r-', linewidth=2, alpha=0.7, label='True Future')
    plt.plot(pred_sample[:, 0], pred_sample[:, 1], 'b--+', linewidth=1.5, alpha=0.7, label='Predicted Path')
    
    plt.plot(0, 0, 'ko', markersize=8, label='Current Position')
    

    for t in range(0, 25, 5):
        if t < len(pred_sample):
            plt.text(pred_sample[t, 0], pred_sample[t, 1], f"{t*0.2:.1f}s", 
                   color='black', fontsize=9, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title(f'Most Likely Trajectory: {mode_name} (p={max_prob:.2f})', fontsize=16)
    plt.xlabel('X Position (Lateral)')
    plt.ylabel('Y Position (Longitudinal)')
    
    plt.xlim(min_x, max_x)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"最可能模式的连续分布表示已保存到 {save_path}")






