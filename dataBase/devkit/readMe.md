CompCars数据集来自网站：http://ai.stanford.edu/~jkrause/cars/car_dataset.html


createTrainaval.py
function:遍历文件夹下面的文件，将其名字集合放到一起

create_vehicle_tf_record_base.py
base文件主要参考/object_detection/dataset_tools/create_pet_tf_record.py
用于生成tf.record文件

create_vehicle_tf_record.py参考create_vehicle_tf_record_base.py
用于生成tf.record文件

cars_test_annos.mat
包含了cars位置坐标信息，详见README.txt
特别提醒cars_test_annos该文件包括narray,比较难于理解，具体操作可参见
create_vehicle_tf_record

其它文件都是通过car_devkit.tgz解压得到

操作手册
python3 createTrainaval.py
------生成train.txt
python3 create_vehicle_tf_record.py
------./output/
            --vehicle_train.record
            --vehicle_test.record
