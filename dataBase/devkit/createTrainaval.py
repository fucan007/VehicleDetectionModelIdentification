import os


source_folder='/home/xiaohui/AI/VehicleDetectionModelIdentification/dataBase/cars_train'#存储图片的目录
dest='./train.txt'#存储train.txt目录
file_list=os.listdir(source_folder)#./image/图片所在路径的文件夹列表
train_file=open(dest,'a')#打开该文件
for file_obj in file_list: #访问文件列表中所有的文件
    file_path=os.path.join(source_folder,file_obj)
    file_name,file_extend=os.path.splitext(file_obj)
    #file_name 保存文件的名字，file_extend保存文件扩展名
    train_file.write(file_name+'\n')#将文件名称写入train_file中并换行
train_file.close()#关闭文件