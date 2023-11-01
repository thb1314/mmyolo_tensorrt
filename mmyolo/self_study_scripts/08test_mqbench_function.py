import torch
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.convert_deploy import convert_deploy
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from self_study_scripts.trt_runner import TRTRunner
import os
import numpy as np
from torchvision.models import resnet50

valdir = '~/dataset/imagenet/imagenet/val/'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=128, shuffle=False, num_workers=24, pin_memory=True)

device = torch.device('cuda')
model = resnet50(pretrained=True).cpu().train()
model_mqbench = prepare_by_platform(model, BackendType.Tensorrt)
enable_calibration(model_mqbench)
model_mqbench.to(device)
model_mqbench.eval()
# calibration loop PTQ process
for data, target in tqdm.tqdm(val_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            model_mqbench(data)
# 下面可以自行定义QAT步骤
enable_quantization(model_mqbench)
model_mqbench.train()

input_shape_dict={'input': [256, 3, 224, 224]}
model_name = "resnet50-mqbench"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
convert_deploy(model_mqbench.eval(), BackendType.Tensorrt, input_shape_dict,
               output_path=output_dir, model_name=model_name)


onnx_filepath = os.path.join(output_dir, model_name + "_deploy_model.onnx")
    
import onnx
def change_input_dim(model):
    sym_batch_dim = "N"
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim                        
def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)
apply(change_input_dim, onnx_filepath, onnx_filepath)

dynamic_range_file = os.path.join(output_dir, model_name + "_clip_ranges.json")
max_batch_size = 128
trt_runner = TRTRunner(onnx_filepath, max_batch_size=max_batch_size, enable_fp16=True, int8_calibrator=None,
                        dynamic_range_file=dynamic_range_file)

def test(model, test_loader):
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in tqdm.tqdm(test_loader):
        # data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data.cpu().numpy())
        output = torch.as_tensor(output['1508'])
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))

test(trt_runner, val_loader)

