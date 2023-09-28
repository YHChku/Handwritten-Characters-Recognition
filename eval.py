import numpy as np # to handle matrix and data operation
import torch # to load pytorch library
from PIL import Image
from network import LeNet5 # load network from network.py
import matplotlib.pyplot as plt
import argparse

#============================ parse the command line =============================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="BestModel.pth", help="pre-trained model")
parser.add_argument("--image", type=str, help="image file")


opt = parser.parse_args()

#============================ start testing =============================================
# build the network
model = LeNet5('SGD')
if torch.cuda.is_available():
    model.cuda()
# load the pre-trained model
model_name = 'checkpoints/' + opt.model
if model_name:
	model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
	print('pretrained model is loaded')
# start testing mode
model.eval()

#======================== image processing =============================================
img_name = 'image/' + opt.image
print('load', img_name)
# read images
img = Image.open(img_name).convert('L')
# crop image as square shape
width, height = img.size
if width > height:
    left = (width - height)/2
    top = 0
    right = width - (width - height)/2
    bottom = height
    img_crop = img.crop((left, top, right, bottom))
else:
    left = 0
    top = (height - width)/2
    right = width
    bottom = height - (height - width)/2
    img_crop = img.crop((left, top, right, bottom))
# resize image to dimension 128*128
img_crop = img_crop.resize((128,128), resample=Image.BICUBIC)
imarray = np.array(img_crop).reshape(1,1,128,128)/255
# convert image as tensor format
if torch.cuda.is_available():
    data = torch.from_numpy(imarray).float().cuda()
else:
    data = torch.from_numpy(imarray).float()
output = model(data)
# obtain the prediction

# digits '0' to '9';
classes = [0x030, 0x031, 0x032, 0x033, 0x034, 0x035, 0x036, 0x037, 0x038, 0x039, 0x041, 0x042, 0x043, 0x044,
                 0x045,
                 0x046, 0x047, 0x048, 0x049, 0x050, 0x051, 0x052, 0x053, 0x054, 0x055, 0x056, 0x057, 0x058, 0x059,
                 0x061,
                 0x062, 0x063, 0x064, 0x065, 0x066, 0x067, 0x068, 0x069, 0x070, 0x071, 0x072, 0x073, 0x074, 0x075,
                 0x076,
                 0x077, 0x078, 0x079, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x5a, 0x6a, 0x6b, 0x6c, 0x6d,
                 0x6e,
                 0x6f, 0x7a]
# sort the output, and then pair up the classes and the percentage
# this command sorts the output in descending order
_, indices = torch.sort(output, descending=True)
# this command calculates the percentage of each class which the input belongs to
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
# this command pairs up the class label and the corresponding percentage
results = [(chr(classes[i]), percentage[i].item()) for i in indices[0][:]]
# print the probability of each class
for i in results:
    print(i)

with open('output.txt', 'w') as txt:
    for i in results:
        txt.write(str(i[0]) + ',' + str(i[1]) + '\n')

# create an empty array for storing the class label
class_label = []
# create an empty array for storing the percentage
percent = []
# open the output.txt in 'r' (read) mode
file = open('output.txt', 'r')
# loop over all the lines inside output.txt
for line in file:
    # split the line by comma. Note that each line has two fields, class label and percentage
    temp = line.split(',')
    # temp[0] represents the class label. We append it to the array called class_label
    class_label.append(temp[0])
    # temp[1] represents the percentage. We append it to the array called percent.
    # Note that we use float(temp[1]) to convert it back into floating number
    percent.append(float(temp[1]))
# after reading the file, we close it
file.close()

# plot the results, x-axis is class_label and y-axis is percent
#plt.plot(class_label, percent)
plt.bar(class_label, percent)
# name the x-axis as 'class label'
plt.xlabel('class label')
# name the y-axis as 'percentage'
plt.ylabel('percentage')
# give the plot a title, 'prediction of "the testing image file name"'
plt.title('prediction of ' + opt.image)
# save the plot to 'output.png' in your current working directory
plt.savefig('output.png')