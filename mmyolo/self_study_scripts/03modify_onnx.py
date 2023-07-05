import onnx
import onnx_graphsurgeon as gs
import mmyolo
import os


def modify_onnx_output(onnx_file, output_onnx_file, add_output_name_list):
    onnx_model = onnx.load(onnx_file)
    gs_onnx_model = gs.import_onnx(onnx_model)
    ori_output = [tensor.name for tensor in gs_onnx_model.outputs]
    real_add_list = []
    # 输出模型每层的输出
    for node in gs_onnx_model.nodes:
        if not (node.outputs[0].name in add_output_name_list):
            continue
        for output_tensor in node.outputs:
            if output_tensor.name not in ori_output and output_tensor.name in add_output_name_list:
                gs_onnx_model.outputs.append(output_tensor)
                real_add_list.append(output_tensor.name)
        
    gs_onnx_model = gs_onnx_model.cleanup().toposort()
    with open(output_onnx_file, 'wb') as f:
        onnx.save(gs.export_onnx(gs_onnx_model), output_onnx_file)

    return real_add_list



mmyolo_dir = os.path.dirname(os.path.dirname(mmyolo.__file__))
yolov6_onnx_dir = os.path.join(mmyolo_dir, 'work_dirs', 'yolov6_n_syncbn_fast_8xb32-400e_coco')
yolov6_onnx_filepath = os.path.join(yolov6_onnx_dir, 'end2end.onnx')
yolov6_onnx_md_filepath = os.path.join(yolov6_onnx_dir, 're_end2end.onnx')

modify_onnx_output(yolov6_onnx_filepath, yolov6_onnx_md_filepath, ['496', 'onnx::Shape_497'])
