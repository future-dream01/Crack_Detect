import os
imgs_path = './images'
count = 0
for name in os.listdir(imgs_path):
    new_name = '*_' + '%05d'%(count) + '.jpg' # *表示类别名
    os.renames(os.path.join(imgs_path, name), os.path.join(imgs_path, new_name))
    os.renames(os.path.join(imgs_path.replace('images', 'xml'), name.replace('jpg', 'xml')), os.path.join(imgs_path.replace('images', 'xml'), new_name.replace('jpg', 'xml')))#同时重命名了对应的xml文件
    count+=1

#注：如果某一个类别的数据有多批，那下一批重命名的时候，则需要记住上一批最后的count数值（可以通过查看上一批数据看count值），然后更改此处的count值。
