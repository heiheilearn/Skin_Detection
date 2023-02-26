import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json

from PIL import Image
from lxml import etree

class VOCDataSet(Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "json_train.txt"):
        self.root = os.path.join(voc_root)
        self.img_root = os.path.join(self.root,"Json_Info", "JPEGImages")
        self.annotations_root = os.path.join(self.root,"Json_Info", "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        print(self.annotations_root)
        print(txt_path)
        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".json")
                        for line in read.readlines() if len(line.strip()) > 0]

        self.xml_list = []
        # check file
        for xml_path in xml_list:
            self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        #print(self.xml_list)

        # read class_indict
        json_file = './pascal_json_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]



        with open(xml_path, encoding="utf-8") as f:
            data = json.load(f)



        img_path = os.path.join(self.img_root, data["imagePath"])
        image = Image.open(img_path)

        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []

        for obj in data["shapes"]:
            #用多边形标注时
            #print(obj)
            if obj['shape_type']=='polygon':
                #print(len(obj['points']))
                label = obj['label']
                xmin=ymin=100000
                xmax=ymax=0
                for point in obj['points']:

                    if point[0]<xmin:
                        xmin=float(point[0])
                    elif point[0]>xmax:
                        xmax=float(point[0])
                    if point[1]<ymin:
                        ymin=float(point[1])
                    elif point[1]>ymax:
                        ymax=float(point[1])

            else:

                label = obj['label']
                xmin = float(obj['points'][0][0])
                xmax = float(obj['points'][1][0])
                ymin = float(obj['points'][0][1])
                ymax = float(obj['points'][1][1])

            print(label, xmin, xmax, ymin, ymax)
            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["label"]])
            iscrowd.append(0)


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path, encoding="utf-8") as f:
            data = json.load(f)

        data_height = int(data["imageHeight"])
        data_width = int(data["imageWidth"])
        return data_height, data_width

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path, encoding="utf-8") as f:
            data = json.load(f)

        data_height = int(data["imageHeight"])
        data_width = int(data["imageWidth"])

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["shapes"]:
            # 用多边形标注时
            # print(obj)
            if obj['shape_type'] == 'polygon':
                # print(len(obj['points']))
                label = obj['label']
                xmin = ymin = 1000
                xmax = ymax = 0
                for point in obj['points']:

                    if point[0] < xmin:
                        xmin = float(point[0])
                    if point[0] > xmax:
                        xmax = float(point[0])
                    if point[1] < ymin:
                        ymin = float(point[1])
                    if point[1] > ymax:
                        ymax = float(point[1])

            else:

                label = obj['label']
                xmin = float(obj['points'][0][0])
                xmax = float(obj['points'][1][0])
                ymin = float(obj['points'][0][1])
                ymax = float(obj['points'][1][1])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["label"]])
            iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

import transforms
from draw_box_utils import draw_objs
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

# read class_indict
category_index = {}
try:
    json_file = open('./pascal_json_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {str(v): str(k) for k, v in class_dict.items()}
    #print(category_index)
    #print(type(category_index))
except Exception as e:
    print(e)
    exit(-1)


data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

# load train data set
train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "json_train.txt")
print(len(train_data_set))

img, target = train_data_set[0]
img = ts.ToPILImage()(img)
plot_img = draw_objs(img,
                    target["boxes"].numpy(),
                    target["labels"].numpy(),
                    np.ones(target["labels"].shape[0]),
                    category_index=category_index,
                    box_thresh=0.5,
                    line_thickness=3,
                    font='arial.ttf',
                    font_size=20)
plt.imshow(plot_img)
plt.show()


for index in random.sample(range(0, len(train_data_set)), k=2):
    img, target = train_data_set[index]
    img = ts.ToPILImage()(img)
    plot_img = draw_objs(img,
                         target["boxes"].numpy(),
                         target["labels"].numpy(),
                         np.ones(target["labels"].shape[0]),
                         category_index=category_index,
                         box_thresh=0.5,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()


'''
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
json_path="./Json_Info/Annotations/正脸-平行偏振光.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
img = cv2.imread('./Json_Info/JPEGImages/正脸-平行偏振光.jpg')

for obj in data['shapes']:
    if obj['shape_type']=='polygon':
        for point in obj['points']:
            print(point)
            cv2.circle(img,(int(point[0]),int(point[1])), 10, (0, 0, 255), -1)


    cv2.imwrite(r"./Json_Info/test1.jpg", img)
'''