# yolo系列模型的部署、精度对齐与int8量化加速


## 一、安装

1. 自定义安装指定版本库(这里可以直接用本仓库中的)

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



```


### 6.3 研究真正没有对齐的原因

topk 在pytorch和onnxruntime上实现不一致

## 7. 动态batch和fp16

```bash

pip install tqdm==4.65.0

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

# onnxruntime
python projects/easydeploy/tools/export.py  $config_file    ${pth_path}      --work-dir work_dirs/$workdir     --img-size 640 640     --batch 1     --device cpu     --simplify      --opset 11       --backend 2     --pre-topk 30000        --keep-topk 300         --iou-threshold 0.65    --score-threshold 0.001


python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --enable-fp16
```

fp16精度

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


## 8. int8量化


tensorrt 7
1. 采用自带Calibrator用TRT内含的QAT算法，
2. 采用set dynamic_range api
全量化

tensorrt8
支持api添加QDQ或者导入onnx QDQ算子
全量化/部分量化
partial quantization


### 8.1 trt7 默认PTQ量化方式

对比 batch=1/4 时 fp32/fp16/int8的精度和latency


```bash

rm ./work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/*.engine  -rf; rm ./calibration.cache;

workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
pth_path=`ls ./work_dirs/$workdir/*.pth`
config_file="work_dirs/$workdir/$workdir.py"

python self_study_scripts/06infer_tensorrt_dynamic_axes_native_ptq.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16
```

默认量化部分数据 默认
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
```


int8 trt7 PTQ 全量训练集数据数据 ENTROPY_CALIBRATION_2
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.493
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
 ```

int8 trt7 PTQ 全量数据 MINMAX_CALIBRATION
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
```

int8 trt7 PTQ 全量数据 MINMAX_CALIBRATION

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
```


### 8.2 trt7 设置dynamic range量化方式 PTQ Mqbench

安装MQbench

```
cd MQBench
pip install -r requirements.txt
python setup.py develop
```

```bash

rm ./work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/*.engine  -rf; rm ./calibration.cache;

python self_study_scripts/07infer_tensorrt_dynamic_axes_mqbench_ptq.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16



python self_study_scripts/09infer_tensorrt_dynamic_axes_mqbench_adaround.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=1 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16
```

MQBench weight 和activation MinMaxObserver 用的数据量比较少
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
 ```

MQBench weight MinMaxObserver activation HistogramObserver 全量数据
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.158
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
```

Adaround advanced PTQ 对传统PTQ方式做了一些更改，比如在fp32->int8 是round还是ceil还是floor

1000 数据
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.380
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.510
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772
```
4000 数据
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.380
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.510
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772
```




分别量化
```
python self_study_scripts/11infer_tensorrt_dynamic_axes_mqbench_minmax.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 &> err.log
```

结果如下
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.380
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.510
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.633
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772
```


去除Adaround 采用全部数据 + 分别量化weight和activation 得到的结果与上面一致，结论就是Adaround没有起作用，分别量化起作用。


**试一下yolov6 small上的表现 和在 yolov8上面的表现**

```bash
workdir="yolov6_s_syncbn_fast_8xb32-400e_coco"
pth_path=`ls ./work_dirs/$workdir/*.pth`
config_file="work_dirs/$workdir/$workdir.py"

mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

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
	--pre-topk 3000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

```

torch version
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.698
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.806
```

tensorrt fp32
```
python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 &> err.log
```

```
cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py 
```

结果
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.698
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.806
```

tensorrt fp16

```
python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --enable-fp16 &> err.log
cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py 
```

结果
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.699
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.806
```


tensorrt int8

```bash
rm ./work_dirs/$workdir/*.engine  -rf;

python self_study_scripts/11infer_tensorrt_dynamic_axes_mqbench_minmax.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=4 \
 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 &> err.log
cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.790
```


```bash
python self_study_scripts/06infer_tensorrt_dynamic_axes_native_ptq.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 &> err1.log
```

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.580
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.683
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.792


yolov8 nano测试

```bash
workdir="yolov8_n_syncbn_fast_8xb16-500e_coco"
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
	--iou-threshold 0.7 \
	--score-threshold 0.001

python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 &> err.log

cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py 

```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.766
```

```bash
python self_study_scripts/05infer_tensorrt_dynamic_axes.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --enable-fp16 &> err.log
cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py 
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.766
```

```bash
python self_study_scripts/06infer_tensorrt_dynamic_axes_native_ptq.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 &> err1.log
mv output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json 
python self_study_scripts/02evel_trt_json_output.py 
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.739
```

```bash
python self_study_scripts/11infer_tensorrt_dynamic_axes_mqbench_minmax.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=1 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 &> err.log
cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.360
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
```


### 8.3 使用pytorch-quantization工具

支持导出trt识别的QDD算子

英伟达TensorRT团队出的，负责插入量化算子，且支持PTQ和QAT，但是QAT支持有限。
并不是基于torch.fx做的，所以需要自己插入量化算子或者用其api自动插入（不好用）

```bash
cd TensorRT/tools/pytorch-quantization
pip install -r requirements.txt
python setup.py develop
```

```python
# 修改 DeConv1D
num_spatial_dims = xxx
```

```bash
workdir="yolov6_s_syncbn_fast_8xb32-400e_coco"
pth_path=`ls ./work_dirs/$workdir/*.pth`
config_file="work_dirs/$workdir/$workdir.py"


python self_study_scripts/12infer_tensorrt_dynamic_axes_nvidia_ptq.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 &> quant_ptq.log


cp output/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.796
```

当采用torch-quantization包来全量化yolov6时，精度出现了问题

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.198
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.685
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.770
```

**采用trex可视化engine**

```bash
cd TensorRT/tools/experimental/trt-engine-explorer
pip install -r requirements.txt
python setup.py develop
```


```bash
trtexec --loadEngine=./output/yolov6_s_syncbn_fast_8xb32-400e_coco_deploy_model.int8.engine --dumpRefit --dumpProfile --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=./output/layer.json --exportProfile=./output/profile.json
```

```bash
python self_study_scripts/11infer_tensorrt_dynamic_axes_mqbench_minmax.py ~/dataset/coco/val2017/ $config_file ./work_dirs/$workdir/end2end.onnx $pth_path --max-batch-size=4 \
 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --out-dir="yolov6-mqbench-ptq" &> err.log
```


```bash
parent_dir='yolov6-mqbench-ptq'
trtexec --loadEngine=./$parent_dir/mqbench_deploy_model.int8.engine --dumpRefit --dumpProfile --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=./$parent_dir/layer.json --exportProfile=./$parent_dir/profile.json
```


## 9. 量化敏感层分析


**map**

```bash
python self_study_scripts/13sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path --max-batch-size=1 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --eval-metric="map" --out-dir="./map_sensitive_analysis" &> map_sensitive_analysis.log

python self_study_scripts/14load_sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path ./map_sensitive_analysis --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --out-dir="./map_sensitive_analysis" &> quant_ptq_map.log
```

精度
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.585
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.690
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.800
```

**mse**

```bash
python self_study_scripts/13sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path --max-batch-size=1 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --eval-metric="mse" --out-dir="./mse_sensitive_analysis" &> mse_sensitive_analysis.log

python self_study_scripts/14load_sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path ./mse_sensitive_analysis --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --out-dir="./mse_sensitive_analysis" &> quant_ptq_mse.log
```

精度
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.590
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.690
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.801
```

**cosine similarity**

```bash
python self_study_scripts/13sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path --max-batch-size=1 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --eval-metric="cosine" --out-dir="./cosine_sensitive_analysis" &> cosine_sensitive_analysis.log

python self_study_scripts/14load_sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path ./cosine_sensitive_analysis --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --out-dir="./cosine_sensitive_analysis" &> quant_ptq_cosine.log
```

精度
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.454
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.690
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.800
```

**sqnr**
```
python self_study_scripts/13sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path --max-batch-size=1 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --eval-metric="sqnr" --out-dir="./sqnr_sensitive_analysis" &> sqnr_sensitive_analysis.log

python self_study_scripts/14load_sensitive_analysis.py ~/dataset/coco/val2017/ $config_file $pth_path ./sqnr_sensitive_analysis --max-batch-size=4 --calib-imgs=./data/coco/train2017 --enable-int8 --enable-fp16 --out-dir="./sqnr_sensitive_analysis" &> quant_ptq_sqnr.log
```

精度
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.799
```


## 10. 很朴素的QAT实现


```bash

workdir="yolov6_s_syncbn_fast_8xb32-400e_coco"
# 表示预加载的PTQ模型
pth_path="./quant_model_rm_sensitive_map/${workdir}_deploy_model.pth"
config_file="work_dirs/$workdir/$workdir.py"

python self_study_scripts/15qat_train_nvidia_trt.py ./data/coco/val2017 $config_file $pth_path   --enable-int8 --enable-fp16 --out-dir="qat_trainning" &> quant_qat.log

quantized_config_file="work_dirs/${workdir}_qat/${workdir}.py"
quantized_model_pth_path="work_dirs/${workdir}_qat/quantized_model.pth"
python self_study_scripts/16qat_export_nvidia_trt.py ./data/coco/val2017 $quantized_config_file $quantized_model_pth_path  --max-batch-size=4 --enable-int8 --enable-fp16 --out-dir="quant_model_rm_sensitive_qat" 

cp -f quant_model_rm_sensitive_qat/result.json self_study_scripts/dynamic_batch_trt_result_int8.json
python self_study_scripts/02evel_trt_json_output.py
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.602
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.694
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.808
```


## 11. 旋转框检测导出onnx和适配


```bash
workdir="rtmdet-r_tiny_fast_1xb8-36e_dota"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
config_file="work_dirs/$workdir/$workdir.py"

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 1024 1024 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.1 \
	--score-threshold 0.05


python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 1024 1024 \
    --device cuda:0 \
	--scale="[[1,3,1024,1024],[1,3,1024,1024],[1,3,1024,1024]]"


python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0

```

