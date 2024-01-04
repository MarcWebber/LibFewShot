import torch
if __name__ == "__main__":
    # model = torch.load(
    #     "F:/study/Grade3.1/machine-learning/LibFewShot/results/deepemd_pretrain_model/miniimagenet/resnet12/max_acc.pth")
    # for name, param in model.items():
    #     print(f'Parameter name: {name}')
    #     print(f'Parameter value:\n{param}\n')

    params = torch.load("./results/deepemd_pretrain_model/miniimagenet/resnet12/max_acc.pth")
    # 为所有参数前加上一个encoder.
    # params = {'encoder.' + k: v for k, v in params.items()}
    torch.save(params, "./results/deepemd_pretrain_model/miniimagenet/resnet12/max_acc_pre.pth")