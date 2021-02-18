import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from thop import profile, clever_format
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    input_test = torch.randn(64, 3, 7, 7).cuda()
    macs, params, = profile(net.model, inputs=([input_test]), verbose=False)
    macs, _ = clever_format([macs, params], "%.3f")
    print('MACs: {}'.format(macs))
    
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt']#, 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    job_done = True

    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3]+'avi')
        vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            #img_h1=imgs.shape[0]
            #img_w1=imgs.shape[1]
            #imgs = imgs[,:] 
            #print(imgs)
            #imgs = cv2.resize(imgs, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            imgs = imgs.cuda()
            with torch.no_grad():
                
                out = net(imgs)
                #torch.Size([1, 3, 288, 800])
                print("imgs.shape",imgs.shape)
                print("color",imgs[0,0,0,0],imgs[0,1,0,0],imgs[0,2,0,0] )
                if not job_done :
                    job_done = True
                    torch.onnx._export(net, imgs, "./ufast_lane_det.onnx", verbose=False,
                    input_names=['input'],output_names=['output1'], 
                    opset_version=12, keep_initializers_as_inputs=True, export_params=True,dynamic_axes=None) 
                
            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]
            #4.0150

            for k in range(len(out)):
                print("out[",k,"].shape",out[k].shape)

            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            #第二个纬度 倒序
            #print("out_j.shape 1",out_j.shape)
            #沿着Z 轴 进行softmax ，每个数 乘以 【1~200]  代表着 图像X 定位的位置。
            #比如 下标 1 ，数值0.9 ，乘以 1 = X分割区域点 1 的位置概率是 0.9
            #下标100 ，数值 0.8，乘以 100 = 分割区域点 100 处，出现概率是 0.8
            #车道最终预测结果取最大，类似一个长的山峰，沿着最高点，选择高处的连线
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(200) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            
            out_j = np.argmax(out_j, axis=0)
            #print("out_j.shape 2",out_j.shape,out_j)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            #print("out_j.shape",out_j.shape,loc)
            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            #out_j (18,4) ,4 条车道，存储x 的位置[0~1]，18 是Y 的序号
            for i in range(out_j.shape[1]):
                #10% 左侧区域开始
                if np.sum(out_j[:, i] != 0) > 1:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            img_h0 = vis.shape[0]
                            img_w0 = vis.shape[1]
                            #print("vis.shape",vis.shape)
                            scalex = img_w0 / 1640 
                            scaley = img_h0 / 590
                            ppp = (int(out_j[k, i] * col_sample_w * img_w * scalex/ 800) - 1,
                             int(img_h * scaley * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            #print("write circle",ppp)
                            cv2.circle(vis,ppp,2,(0,255,0),-1)
            vout.write(vis)
            cv2.imshow('imshow',vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        vout.release()