#导出模型
python3 export_inference_graph.py \
--model_name=cifarnet \
--batch_size=1 \
--dataset_name=cifar10 \
--output_file=cifarnet_graph_def.pb \
--dataset_dir=./cifar10/

#冻结模型
python3 freeze_graph.py \
--input_graph=cifarnet_graph_def.pb \
--input_binary=true \
--input_checkpoint="./cifarnet-model/model.ckpt-100000" \
--output_graph=freezed_cifarnet.pb \
--output_node_names=output