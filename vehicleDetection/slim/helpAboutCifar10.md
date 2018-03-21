关于cifar10数据级
从下载数据－－－转换数据－－－－训练－－－－验证－－－－导出模型－－－－冻结模型－－－－用一张图片测试

该脚本负责数据的下载转换训练以及验证
位置./script下面
train_cifarnet_on_cifar10.sh

注意这俩个环境变量，不同环境下需要更改
TRAIN_DIR　　　　存储数据
DATASET_DIR　　　存储模型
另外需要注意环境是否支持gpu
如果不支持需要修改clone_on_cpu true
当train_cifarnet_on_cifar10.sh执行完毕
slim/cifar10        会有相应的下载好的数据生成:cifar10_test.tfrecord  cifar10_train.tfrecord  labels.txt 由于该文件超过了100M所以不往github上上传了
slim/cifarnet-model　训练生成的模型checkpoint  model.ckpt-100000.data-00000-of-00001  model.ckpt-100000.index  model.ckpt-100000.meta
                    只保留了这些剩下的已经删除


当export_inference_graph_cifar10.sh执行完毕
cifarnet_graph_def.pb   导出的模型
freezed_cifarnet.pb     冻结的模型


test_cifar10.sh选举一张图片进行测试

结果如下：
id:[2] name:[bird] (score = 0.74933)
id:[3] name:[cat] (score = 0.09537)
id:[4] name:[deer] (score = 0.09519)
id:[0] name:[airplane] (score = 0.02756)
id:[1] name:[automobile] (score = 0.01199)

