import torch
from argparse import ArgumentParser
from mmengine.config import Config, DictAction
import os.path as osp
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from self_study_scripts.utils import de_parallel, deepcopy
from self_study_scripts.utils import model_quant_enable, model_calib_disable, model_quant_disable
import os


def save_calib_model(model, output_model_path):
    # Save calibrated checkpoint
    print('Saving calibrated model to {}... '.format(output_model_path))
    dirname = os.path.dirname(output_model_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    torch.save({'model': deepcopy(de_parallel(model))}, output_model_path)
    

def load_checkpoint(weights, map_location=None):
    """Load model from checkpoint file."""
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    model = model.eval()
    return model



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'val-img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('torch_checkpoint', help='Pytorch Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--enable-fp16', action='store_true', default=False, help='Whether enable tensorrt fp16 inference')
    parser.add_argument(
        '--enable-int8', action='store_true', default=False, help='Whether enable tensorrt int8 inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

# QAT步骤
# 1. 加载PTQ的校准预训练(采用什么方法？)
# 2. 采用一个较小学习率在此基础上微调(有没有蒸馏、量化参数是否参与训练)
# 3. 导出 onnx

def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)

    model = load_checkpoint(args.torch_checkpoint, map_location='cpu')
 
    # work_dir is determined in this priority: CLI > segment in file > filename
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 先采用较强数据增强，在最后num_last_epochs里采用较弱数据增强

    # 修改 epoch
    cfg.work_dir = cfg.work_dir + '_qat'
    cfg.max_epochs = 1
    cfg.load_from = None
    cfg.num_last_epochs = 1
    # 0.1 * (lr * 0.01)
    cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr * 0.001
    cfg.optim_wrapper.optimizer.batch_size_per_gpu = 20
    cfg.train_cfg.max_epochs = 30
    cfg.train_cfg.val_interval = 1
    cfg.train_cfg.dynamic_intervals = [(0, 30)]

    # 去除EMA Hook
    cfg.custom_hooks.pop(0)
    # switch epoch 改成0 表示一开始就切换到 较弱数据增强 训练阶段
    cfg.custom_hooks[0].switch_epoch = 0
    
    # 去除自动化调参
    cfg.default_hooks.pop('param_scheduler')
    
    cfg.train_dataloader.batch_size = 16
    
    # 去除warm up lr schedule
    cfg.model.train_cfg.initial_epoch = 0

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)


    # 修改 model
    def modify_runner_model(runner, model: torch.nn.Module):
        
        runner_model: torch.nn.Module = runner.model
        # runner.model = model

        model2dict = dict()
        for name, module in model.named_children():
            model2dict[name] = module
        
        # no_param_dict = dict()
        # for name, module in runner_model.named_children():
        #     if 'bbox_head' == name:
        #         bbox_head = module
        #         for name, module in bbox_head.named_children():
        #             if len(list(module.parameters())) == 0:
        #                 no_param_dict[name] = module
        #         break

        for name, module in runner_model.named_children():
            if len(list(module.parameters())) > 0:
                device = next(module.parameters()).device
                if name in model2dict:
                    runner_model.add_module(name, model2dict[name].to(device))
                    print(f'{name} is replaced')


        # for name, module in runner_model.named_children():
        #     if 'bbox_head' == name:
        #         bbox_head = module
        #         for name, module in bbox_head.named_children():
        #             if len(list(module.parameters())) == 0:
        #                 bbox_head.add_module(name, no_param_dict[name])
        #         break
            
        # set inited
        for name, module in runner_model.named_modules():
            module._is_init = True

    model_quant_enable(model)
    model_calib_disable(model)
    modify_runner_model(runner, model)
    
    print('runner.model')
    print(runner.model)

    # start training
    try:
        runner.val()
    except:
        pass
    runner.train()
    save_calib_model(runner.model, os.path.join(cfg.work_dir, 'quantized_model.pth'))

if __name__ == '__main__':
    main()
