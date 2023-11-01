workdir="rtmdet-r_tiny_fast_1xb8-36e_dota"
pth_path=`ls ./work_dirs/$workdir/*.pth`
config_file="work_dirs/$workdir/$workdir.py"

# python projects/easydeploy/tools/export.py \
# 	work_dirs/$workdir/$workdir.py \
# 	${pth_path} \
# 	--work-dir work_dirs/$workdir \
#     --img-size 1024 1024 \
#     --batch 1 \
#     --device cpu \
#     --simplify \
# 	--opset 11 \
# 	--backend 2 \
# 	--pre-topk 1000 \
# 	--keep-topk 200 \
# 	--iou-threshold 0.1 \
# 	--score-threshold 0.05

# pushd /home/thb/tmp/mmyolo-hb/mmyolo/self_study_scripts/efficientNMSPluginForRotateBox
# make
# popd

# python projects/easydeploy/tools/build_engine.py      ./work_dirs/$workdir/end2end.onnx  --scale="[[1,3,1024,1024],[1,3,1024,1024],[1,3,1024,1024]]"   --img-size 1024 1024     --device cuda:0

python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0
    
# python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
#     ~/dataset/coco/val2017/ \
#     $config_file \
#     ./work_dirs/$workdir/end2end.engine \
#     $pth_path \
#     --device cuda:0
