import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm


index = 7
name = 'mv_pn_mi_2'
path = ['sketch_feat.npy','model_feat.npy' ]

p_q = path[0] # #path1[index]
p_c = path[1] #path2[index]


def get_data(p_q, p_c):
    #a = loadmat(p_q)
    #b = loadmat(p_c)
    a = np.load(p_q,allow_pickle=True).item()
    b = np.load(p_c,allow_pickle=True).item()
    #print('a keys:', a.keys())
    lens = -1 
    fts_q = a['fts']  # feature qurey
    las_q = a['las']  # label
    fts_c = b['fts']  # feature contrain
    las_c = b['las']  # label
    #print(fts_q.shape, fts_c.shape)
    las_q = np.array(las_q).reshape(-1)  # 规范标签的形状
    las_c = np.array(las_c).reshape(-1)
    return fts_q,las_q, fts_c,las_c

# 输入q,c batch x feature
# 输出：q * c  --query x contain
def dist_euler(fts_q, fts_c):
    fts_qs = np.sum(np.square(fts_q),axis=-1,keepdims=True)
    fts_cs = np.sum(np.square(fts_c),axis=-1,keepdims=True).T
    qc = np.matmul(fts_q,fts_c.T)
    dist = fts_qs + fts_cs - 2 * qc
    return dist

def dist_cos(fts_q, fts_c):
    
    up = np.matmul(fts_q,fts_c.T)
    down1 = np.sqrt(np.sum(np.square(fts_q),axis=-1,keepdims=True))
    down2  = np.sqrt(np.sum(np.square(fts_c),axis=-1,keepdims=True).T)
    down = np.matmul(down1, down2)
    dist = up/(down+1e-4)
    return 1-dist


def get_pr(dist_p=0):
    fts_q,las_q, fts_c,las_c = get_data(p_q, p_c)
    if dist_p==0:
        dist = dist_euler(fts_q, fts_c)
    else:dist = dist_cos(fts_q, fts_c)
 
    dist_index = np.argsort(dist, axis=-1)
    dist = np.sort(dist, axis=-1)

    len_q,len_c = dist.shape
    
    cls_num = np.sort(np.unique(las_c))
    num_per_cls = np.array([ np.sum(las_c==i) for i in cls_num])  # 目标域各类的数目
    C = num_per_cls[las_q]
    P_points = np.zeros((len_q, C.max()))
    Av_Precision = np.zeros(len_q)
    NN = np.zeros(len_q)
    FT = np.zeros(len_q)
    ST = np.zeros(len_q)
    dcg = np.zeros(len_q)
    E = np.zeros(len_q)
    result = np.zeros_like(dist)
    
    filename = 'Stats.txt'
    fid = open(filename,'w') 
    fid.write('        NN     FT     ST      E       DCG\n')
    fid.close()

    
    laq_bool = np.tile(las_q,(len_c,1)).T
    lac_bool = np.tile(las_c,(len_q,1))  # 需要对c 的标签进一步排序
    index = np.tile(np.array(range(len_q)),(len_c,1)).T
    lac_bool = lac_bool[index, dist_index]
    result = (laq_bool == lac_bool)  # 得到了每个对象的检索结果
    
    G_sum = result.cumsum(axis=-1)
    #P_points = []
    
    for i,rs in enumerate(result):   # G=rs
        item = (np.arange(C[i])+1)/(np.arange(len(rs))[rs.astype(bool)]+1)
        P_points[i, :len(item)] = item
        Av_Precision[i] = item.mean()

        NN[i] = item[0]
        FT[i] = G_sum[i,C[i]]/C[i]
        ST[i] = G_sum[i,2*C[i]]/C[i]
        # calculate E
        P_32 = G_sum[i, 32]/32
        R_32 = G_sum[i, 32]/C[i]
        
        if P_32==0 and R_32==0:
            E[i]=0
        else:
            E[i]=2* P_32 * R_32/(P_32 + R_32)
        # calculate dcg
        NORM_VALUE = 1 + np.sum(1/np.log2(np.arange(1,C[i])+1))
        dcg_i = (1/np.log2(np.arange(1, len_c)+1)) * rs[1:]
        dcg_i = rs[0] + dcg_i.sum()
        dcg[i] = dcg_i/NORM_VALUE

        filename = 'Stats.txt'
        fid = open(filename, 'a')
        fid.write('No.%d: %2.3f\t %2.3f\t %2.3f\t %2.3f\t %2.3f\n' % (i, NN[i],FT[i],ST[i],E[i],dcg[i]))
        fid.close()
    
    NN_av = NN.mean()
    FT_av= FT.mean()
    ST_av = ST.mean()
    dcg_av = dcg.mean()
    E_av = E.mean()
    Mean_Av_Precision = Av_Precision.mean()
    
    filename = 'Stats.txt'
    fid = open(filename, 'a')
    fid.write('        NN     FT     ST      E       DCG\n')
    fid.write('%2.3f\t %2.3f\t %2.3f\t %2.3f\t %2.3f\n' % (NN_av,FT_av,ST_av,E_av,dcg_av))
    print('%2.3f\t %2.3f\t %2.3f\t %2.3f\t %2.3f' % (NN_av,FT_av,ST_av,E_av,dcg_av))
    print('mAP:%2.3f' % (Mean_Av_Precision))
    fid.close()

    filename = 'PR_test.txt'
    #fid = open(filename, 'w')
    #fid.close()
    calcAvgPerf(P_points, C, len_q, filename)

    return 0,0,0


def calcAvgPerf(P_points, C, size, filename):
    CUTOFF = 2
    SAMPLE = 20
    
    mean = np.zeros(SAMPLE)

    for j in range(SAMPLE):
        valid = 0
        for i in range(size):
            if C[i] < CUTOFF or C[i] < (SAMPLE/(j+1)):
                continue
            temp = interpolatePerf(P_points[i,:], C[i], (j+1)/SAMPLE)
            mean[j] = mean[j] + temp
            valid = valid + 1


        if valid > 0:
            mean[j]=mean[j]/valid
    pr = np.array([(np.arange(SAMPLE)+1)/SAMPLE,mean])
    fo = open(filename,'w')
    #print('----', filename)
    np.savetxt(fo, pr,fmt='%.5f')
    fo.close()
    return pr 
        

def interpolatePerf(value, num, index):
    bins = num

    if bins == 0:
        v = 0
    else:
        xx = index * bins
        x1 = np.fix(xx)
        x2 = x1+1
        dx = xx-x1
        if x1<1: x1=1
        if x2<1: x2=1

        if x1>bins: x1 = bins
        if x2>bins: x2 = bins
        #print('----',x1,x2,value.shape) 
        v = value[int(x1-1)]*(1-dx)+ value[int(x2-1)]*dx
        return v




def get_pr_cls(cls, dist_p=0):
    fts_q,las_q, fts_c,las_c = get_data(p_q, p_c)
    fts_q = fts_q[las_q==cls]
    las_q = np.array([cls]).repeat(len(fts_q))
    if dist_p==0:
        dist = dist_euler(fts_q, fts_c)
    else:dist = dist_cos(fts_q, fts_c)
    
    dist_index = np.argsort(dist, axis=-1)
    dist = np.sort(dist, axis=-1)


    # 利用标签计算，标记检索结果
    len_q,len_c = dist.shape
    result = np.zeros_like(dist)
    laq_bool = np.tile(las_q,(len_c,1)).T
    lac_bool = np.tile(las_c,(len_q,1))  # 需要对c 的标签进一步排序
    index = np.tile(np.array(range(len_q)),(len_c,1)).T
    lac_bool = lac_bool[index,dist_index]
    result = (laq_bool == lac_bool)

    p = np.zeros(len_c)
    r = np.zeros(len_c)
    r_all = np.sum(result)
    for i in range(len_c):
        s = np.sum(result[:,:i+1])
        p[i] = s/((i+1)*len_q)
        r[i] = s/r_all
    mAP = np.sum((r[1:] - r[:-1])*p[:-1])
    mAP5 = p[:5].mean()
    fo =open('pr/%s_%s_%.3f_%.3f.txt' %(name, 'cos' if dist_p else 'elur',mAP,mAP5),'w') 
    np.savetxt(fo, np.array([r, p]), fmt='%.5f')
    fo.close() 
    return np.array([r, p]),mAP,mAP5



if __name__ == '__main__':

    pr,mAP,mAP5 = get_pr(0)
    #print('elur map:',mAP,',',mAP5) 
    
    pr,mAP,mAP5 = get_pr(1)
    #print('cos map:',mAP,',',mAP5) 
    #a = []
    #for i in range(90):
    #    pr,mAP,mAP5 = get_pr_cls(i,1)
    #    a.append(mAP)
    #    print('cls:%d,  mAP:%.4f, mAP5:%.4f'%(i,  mAP, mAP5))
    #np.save('cls_mAP.npy', np.array(a)) 
    # print('pr:',pr.shape)
    # plt.plot(pr[0,:],pr[1,:])
    # plt.title('map:%f'%map)
    # plt.show()
