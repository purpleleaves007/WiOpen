import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
#import cv2

class datatrcsi(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0
        data_listdfs = r'D:\Data\Widar3\STIDFM'
        #data_listdfs = r'D:\Data\Widar3\DFSra'

        for i in [0,1,2,3,4,5,6,7,8]:
            for j in range(1,5):
                for k in range(1,6):
                    for o in range(1,6):
                        for n in [1,2,3,4]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.n_data = self.n_data +1
        for i in [0,1,2,3,4,5,6,7,8]:
            for j in range(1,4):
                for k in range(1,6):
                    for o in range(1,6):
                        for n in [1,2,3,4]:
                            file = data_listdfs + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_pathsdfs.append(file)

     def __getitem__(self, item):
        transform1 =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths, img_pathsdfs, labels = self.img_paths[item], self.img_pathsdfs[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        dfs = Image.open(img_pathsdfs)#.convert('L')
        dfs = dfs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            dfs = transform1(dfs)
            labels = int(labels)
        #print(self.n_data)
        return dfs, inputs, labels, item

     def __len__(self):
        return self.n_data

class datatecsi(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        count = 0
        self.n_data = count

        for i in [0,1,2,3,4,5,6,7,8]:
            for j in range(1,4):
                for k in range(1,6):
                    for o in range(1,6):
                        for n in [5]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.n_data = self.n_data +1

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths, label = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            label = int(label)

        return inputs, inputs, label, item

     def __len__(self):
        return self.n_data
     
class datatecsiun(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        count = 0
        self.n_data = count

        for i in range(0,9):
            for j in range(4,10):
                for k in range(1,6):
                    for o in range(1,6):
                            for n in [5]:
                                files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                                self.img_paths.append(files)
                                self.img_labels.append(3)
                                self.n_data = self.n_data +1

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths, label = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            label = int(label)

        return inputs, inputs, label, item

     def __len__(self):
        return self.n_data

class datatrcsip(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0
        data_listdfs = r'D:\Data\Widar3\STIDFM'
        #data_listdfs = r'D:\Data\Widar3\DFSra'

        for i in range(0,6):
            for j in range(1,7):
                for k in [2,1,4,5]:
                    for o in range(1,6):
                        for n in range(1,5):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.n_data = self.n_data +1
        for i in range(0,6):
            for j in range(1,7):
                for k in [2,1,4,5]:
                    for o in range(1,6):
                        for n in range(1,5):
                            file = data_listdfs + '/' + str(i+1) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_pathsdfs.append(file)

     def __getitem__(self, item):
        transform1 =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths, img_pathsdfs, labels = self.img_paths[item], self.img_pathsdfs[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        dfs = Image.open(img_pathsdfs)#.convert('L')
        dfs = dfs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            dfs = transform1(dfs)
            labels = int(labels)

        return dfs, inputs, labels, item

     def __len__(self):
        return self.n_data

class datatecsip(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0

        for i in range(0,6):
            for j in range(1,7):
                for k in [3]:
                    for o in range(1,6):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.n_data = self.n_data +1

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths,  labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, inputs, labels, item

     def __len__(self):
        return self.n_data
     
class datatecsipun(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0

        for i in range(0,9):
            for j in range(1,2):
                for k in [1,2,3,4,5]:
                    for o in range(1,6):
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(6)
                            self.n_data = self.n_data +1

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths,  labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, inputs, labels, item

     def __len__(self):
        return self.n_data

class datatrcsio(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0
        data_listdfs = r'D:\Data\Widar3\STIDFM'
        #data_listdfs = r'D:\Data\Widar3\DFSra'

        for i in range(0,6):
            for j in range(1,7):
                for k in range(1,6):
                    for o in [1,3,4,5]:
                        for n in range(1,5):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.n_data = self.n_data +1
        for i in range(0,6):
            for j in range(1,7):
                for k in range(1,6):
                    for o in [1,3,4,5]:
                        for n in range(1,5):
                            file = data_listdfs + '/' + str(i+1) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_pathsdfs.append(file)

     def __getitem__(self, item):
        transform1 =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths, img_pathsdfs, labels = self.img_paths[item], self.img_pathsdfs[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        dfs = Image.open(img_pathsdfs)#.convert('L')
        dfs = dfs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            dfs = transform1(dfs)
            labels = int(labels)

        return dfs, inputs, labels, item

     def __len__(self):
        return self.n_data

class datatecsio(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0

        for i in range(0,6):
            for j in range(1,7):
                for k in range(1,6):
                    for o in [2]:
                        for n in [1,2,3,4,5]:
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(j-1)
                            self.n_data = self.n_data +1

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths,  labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, inputs, labels, item

     def __len__(self):
        return self.n_data
     
class datatecsioun(data.Dataset):
     def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_pathsdfs = []
        self.img_labels = []
        self.n_data = 0

        for i in range(0,9):
            for j in range(7,10):
                for k in range(1,6):
                    for o in [1,2,3,4,5]:
                        for n in range(1,6):
                            files = data_list + '/' + str(i) + '-' + str(j) + '-' + str(k) + '-' + str(o) + '-' +str(n) +'.jpg'
                            self.img_paths.append(files)
                            self.img_labels.append(6)
                            self.n_data = self.n_data +1

     def __getitem__(self, item):
        transform =  transforms.Compose([
                                            transforms.ToTensor(), # 转化为pytorch中的tensor
                                            ]) # 主要改这个地方

        img_paths,  labels = self.img_paths[item], self.img_labels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = transform(inputs)
            labels = int(labels)

        return inputs, inputs, labels, item

     def __len__(self):
        return self.n_data