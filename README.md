# 基于mmyolo框架探索yolo系列模型的TensorRT部署与精度对齐

## 一、安装

1. 自定义安装指定版本库

```python

git clone -b 'v0.5.0' https://github.com/open-mmlab/mmyolo.git --single-branch mmyolo

git clone -b 'v0.7.2' https://github.com/open-mmlab/mmengine.git --single-branch mmengine

git clone -b 'v3.0.0rc6' https://github.com/open-mmlab/mmdetection.git --single-branch mmdetection

git clone -b 'v2.0.0rc4' https://github.com/open-mmlab/mmcv.git --single-branch mmcv

conda create -n mmyolo_zl python=3.8

# CUDA 10.2
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

# 切换到代码目录
wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz

tar -zxvf onnxruntime-linux-x64-1.12.1.tgz
cd onnxruntime-linux-x64-1.12.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

pip install onnx==1.11.0
pip install onnxruntime==1.12.1
pip install onnxsim==0.4.8
pip install opencv-python==4.7.0.72
pip install yapf==0.32.0
pip install pyyaml==6.0


cd mmengine
pip install -v -e .
cd ..

cd mmcv
# 编译安装mmcv相关onnxruntime算子
MMCV_WITH_OPS=1 python setup.py develop
cd ..

cd mmdetection
pip install -v -e .
cd ..

# mmrotate
git clone -b 'v1.0.0rc1' https://github.com/open-mmlab/mmrotate.git --single-branch mmrotate
cd mmrotate
pip install -v -e . 
cd ..

cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
pip install -v -e .

# downgrade  protobuf package version 3.20.x
pip install protobuf==3.20.0

pip install netron

# install tensorrt
pip install pycuda==2022.1
# 到自己的tensorrt下面找到whl包并安装

pip install numpy==1.21.6
```

## 2. 初体验

https://mmyolo.readthedocs.io/zh_CN/dev/get_started/15_minutes_object_detection.html#easydeploy

## 3. 导出yolo系列模型

```bash

# 导出onnx tensorrt版本
# yolov5
python projects/easydeploy/tools/export.py \
	configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
	work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
	--work-dir work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25


# yolov6
pip install openmim==0.3.7

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 30000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

# 测试精度， 为加快速度还请去掉config文件中的 visualization hook
# python tools/test.py work_dirs/$workdir/$workdir.py \
#                      $pth_path \
# 					 --json-prefix scripts/yolov6_n_coco_result


# yolov7
workdir="yolov7_tiny_syncbn_fast_8x16b-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25


# yolov8
workdir="yolov8_n_syncbn_fast_8xb16-500e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25


# yolox
workdir="yolox_tiny_fast_8xb8-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 3 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25


# ppyoloe
workdir="ppyoloe_plus_s_fast_8xb8-80e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25

# rtmdet
workdir="rtmdet_tiny_syncbn_fast_8xb32-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25



# rtmdet 旋转框

workdir="rtmdet-r_tiny_fast_1xb8-36e_dota"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir


pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25


```

## 4. 初步导出TensorRT

```bash

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

python projects/easydeploy/tools/export.py \
	$config_file \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.3

python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0

python projects/easydeploy/tools/image-demo.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    --device cuda:0
```


## 5. TensorRT模型精度的验证

```bash

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

python projects/easydeploy/tools/export.py \
	$config_file \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 30000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0


python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0

python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    ~/dataset/coco/val2017/ \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0
 

```

目标：
1. 探索相同输入下tensorrt和pytorch两个框架的输出有无不同
2. 针对 coco-val 数据集下的精度，tensorrt和pytorch得到的有何区别？


得到yolov6 pytorch版本输出

```bash
# 测试精度， 为加快速度还请去掉config文件中的 visualization hook
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
pth_path=`ls ./work_dirs/$workdir/*.pth`
python tools/test.py work_dirs/$workdir/$workdir.py \
                     $pth_path \
					 --json-prefix scripts/yolov6_n_coco_result
```


## 6. 精度没有百分之百对齐的原因

### 6.1 仔细研究pytorch后处理和TensorRT后处理（看onnx和源码）

定位问题：
解码后的score和boxes完全一致，定位到nms后处理部分不一致。
猜测原因：
1. pytorch中有些框在取topk的过程当中被过滤掉了，比如其分值满足>score_threshold, 但是不在topk当中
2. pytorch和tensorrt有些算子在实现上不一致

topk的不一致
nms算子与torch的nms可能也不一致

### 6.2 尝试对齐这种后处理

```bash

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

# onnxruntime
python projects/easydeploy/tools/export.py  $config_file    ${pth_path}      --work-dir work_dirs/$workdir     --img-size 640 640     --batch 1     --device cpu     --simplify      --opset 11       --backend 5     --pre-topk 30000        --keep-topk 300         --iou-threshold 0.65    --score-threshold 0.001


python self_study_scripts/03modify_onnx.py

python self_study_scripts/01cmp_output_with_torch_tensorrt.py     ~/dataset/coco/val2017/     $config_file     ./work_dirs/$workdir/re_end2end.onnx     $pth_path     --device cuda:0 &> err.log


# tensorrt
python projects/easydeploy/tools/export.py  $config_file    ${pth_path}      --work-dir work_dirs/$workdir     --img-size 640 640     --batch 1     --device cpu     --simplify      --opset 11       --backend 5     --pre-topk 3840        --keep-topk 300         --iou-threshold 0.65    --score-threshold 0.001

python self_study_scripts/03modify_onnx.py

python projects/easydeploy/tools/build_engine.py      ./work_dirs/$workdir/re_end2end.onnx     --img-size 640 640     --device cuda:0


python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    ~/dataset/coco/val2017/ \
    $config_file \
    ./work_dirs/$workdir/re_end2end.engine \
    $pth_path \
    --device cuda:0

# note 把配置文件中nms_pre=30000 参数改为nms_pre=3840
python self_study_scripts/01cmp_output_with_torch_tensorrt.py     ~/dataset/coco/val2017/     $config_file     ./work_dirs/$workdir/re_end2end.engine     $pth_path     --device cuda:0 --result-json-path=self_study_scripts/trt-1.json &> err.log


python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4

python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --enable-fp16

```


### 6.3 研究真正没有对齐的原因

topk 在pytorch和onnxruntime上实现不一致

## 7. 动态batch和fp16

```bash

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

# onnxruntime
python projects/easydeploy/tools/export.py  $config_file    ${pth_path}      --work-dir work_dirs/$workdir     --img-size 640 640     --batch 1     --device cpu     --simplify      --opset 11       --backend 2     --pre-topk 30000        --keep-topk 300         --iou-threshold 0.65    --score-threshold 0.001

```

## 8. int8量化

对比 batch=1/4 时 fp32/fp16/int8的精度和latency


fp16
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.771
```

## 9. 旋转框检测导出onnx


## 10. ...


