import os
from tensorflow.python import pywrap_tensorflow
 
# code for finall ckpt
# checkpoint_path = os.path.join('~/tensorflowTraining/ResNet/model', "model.ckpt")
 
# code for designated ckpt, change 3890 to your num
checkpoint_path = os.path.join('/home/hadoop/python_project/mycode/snapshot', "iter_12500.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
model={}
# Print tensor name and values
for key in var_to_shape_map:
    npy_out={}
    #print("tensor_name: ", key)
    #print(reader.get_tensor(key))
    str_name=key
    if str_name.find('Adam')>-1:
        continue
    if str_name.find('/')>-1:
        names=str_name.split('/')
        length=range(len(names))
        assert length==2,'name深度超过2'
        model[names[0]][names[1]]=reader.get_tensor(key)
    else:
        model[names]=reader.get_tensor(key)
    np.save('model.npy',model)


            
        
