import onnx_graphsurgeon as gs
import onnx
import os
# 对所有的输入节点的graph过滤掉重复的qdq节点

def merge_common_qdq(onnx_filepath, output_filepath = None):
    onnx_model = onnx.load(onnx_filepath)

    gs_model = gs.import_onnx(onnx_model)

    input2nodes = dict()
    output2nodes = dict()
    name2node = dict()

    for node in gs_model.nodes:
        name2node[node.name] = node
        for input_name in node.inputs:
            input2nodes.setdefault(input_name.name, set()).add(node.name)
        for output_name in node.outputs:
            output2nodes.setdefault(output_name.name, set()).add(node.name)

    tmp_dict1 = dict()
    for k in input2nodes:    
        if k in output2nodes:
            tmp_dict1[k] = [name2node[item] for item in output2nodes[k]]

    tmp_dict2 = dict()
    for k in output2nodes:
        if k in input2nodes:
            tmp_dict2[k] = [name2node[item] for item in input2nodes[k]]

    input2nodes = tmp_dict1
    output2nodes = tmp_dict2

    def get_next_node(node):
        next_node = None
        try:
            return node.o(0)
        except:
            return next_node

    next_node_is_dq = lambda x: get_next_node(x) is not None and get_next_node(x).op == "DequantizeLinear"

    for node in gs_model.nodes:
        for output_tensor in node.outputs:
            output_name = output_tensor.name
            if output_name not in output2nodes:
                continue
            output_nodes = output2nodes[output_name]
            if len(output_nodes) <= 1:
                continue
            is_qdq = output_nodes[0].op == "QuantizeLinear" and next_node_is_dq(output_nodes[0])
            for output_node in output_nodes[1:]:
                is_qdq &= output_node.op == "QuantizeLinear" and next_node_is_dq(output_node)
                if not is_qdq:
                    continue
            if not is_qdq:
                continue

            keep_dq_node = output_nodes[0].o(0)
            for output_node in output_nodes[1:]:
                output_node.inputs.clear()
                dq_node = output_node.o(0)
                dq_next_node = dq_node.o(0)
                dq_output_tensor = dq_node.outputs[0]
                
                dq_node.outputs.clear()
                for tmp_node in output2nodes[dq_output_tensor.name]:
                    for i, input_tensor in enumerate(tmp_node.inputs):
                        if dq_output_tensor == input_tensor:
                            tmp_node.inputs[i] = keep_dq_node.outputs[0]

    gs_model.cleanup().toposort()
    output_filepath = output_filepath or os.path.join(os.path.dirname(onnx_filepath), 're_' + os.path.basename(onnx_filepath))
    onnx.save(gs.export_onnx(gs_model), output_filepath)
