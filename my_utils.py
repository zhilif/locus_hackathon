import torch
import os
import clip
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from tqdm.auto import tqdm
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
import pickle
import random
import re
import math
import sys
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from operator import itemgetter
from elasticnet import glm_saga, IndexedTensorDataset
from sklearn import linear_model

cifar_classes = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

image_templates = [
    'a photo of {}.'
]

image_size=224
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size,interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])


def set_randomness(rand_seed=0):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_feature_text(filename):
    gpt3_responses = pickle.load(open(filename, 'rb'))
    feature_text= []
    for k, rep in gpt3_responses.items():
        s1 = rep['choices'][0]['text']
        arr = s1.split('\n')
        arr = list(filter(lambda x: len(x)>0, arr))
        for item in arr:
            feature_text.append(re.sub('[0-9]*\. |- |• |-|•', '', item).lower().strip())

    return feature_text

def filter_dup(train_X, threshold=0.95):
    corr = torch.corrcoef(train_X.T.float())
    print(corr.shape)
    dup = set()
    for i in range(len(corr)):
        if(torch.any(corr[i, 0:i]>threshold)):
#             print(i)
            dup.add(i)

    return [i for i in range(len(corr)) if i not in dup]

def get_feature_text_with_classnames(filename, classnames):
    feature_text = get_feature_text(filename)
    clip_prompts = []
    for classname in cifar_classes:
        clip_prompts.extend([template.format(classname) for template in image_templates])
    feature_text.extend(clip_prompts)
    return feature_text

def get_prompt_encodings(model, feature_text, device='cuda:0', bsz=100):
    encodings = []
    with torch.no_grad():
        num_batches = math.ceil(len(feature_text)/bsz)
        for batch in tqdm(range(num_batches)):
            prompts = feature_text[batch*bsz:(batch+1)*bsz]
            text_features = model.encode_text(clip.tokenize(prompts).to(device))
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            encodings.append(text_features)
    return torch.cat(encodings)

def clip_forward(model, image, text_features):
    image_features = model.encode_image(image)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text, image_features

class NPZDataset(Dataset):
    def __init__(self, path, key='x', use_files=None, transform=None):
        self.path = path
        self.files = []
        if use_files:
            self.files = use_files
        else:
            self.files = list(Path(path).glob('*.npz'))
        self.transform = transform
        self.key = key
    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = np.load(str(self.files[item]))
        x = torch.from_numpy(data[self.key])
        y = torch.from_numpy(data['y'])
        if self.transform is not None:
            x = self.transform(x)
        return x, y



def load_npz_img(folder, counter=None):
    img_features = []
    Y = []
    for file in tqdm(list(Path(folder).glob('*.npz'))):
        if counter and len(Y) >= counter:
            break
        data = np.load(file)
        img_features.append(torch.from_numpy(data['img']).float())
        Y.append(torch.from_numpy(data['y']).long())
    img_features=torch.cat(img_features)
    Y = torch.cat(Y)
    return img_features, Y

def load_npz_prompts(folder, counter=None):
    X = []
    Y = []
    for file in tqdm(list(Path(folder).glob('*.npz'))):
        if counter and len(Y) >= counter:
            break
        data = np.load(file)
        X.append(torch.from_numpy(data['x']).float())
        Y.append(torch.from_numpy(data['y']))
    X=torch.cat(X)
    Y = torch.cat(Y)
    return X, Y
# repo_root = os.path.join(os.getcwd(), './CIFAR101/code')
# sys.path.append(repo_root)
#
# from CIFAR101.code import utils

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, x_tensors, y_tensors=None, transform=None):
        self.has_y = False
        if y_tensors is not None:
            assert len(x_tensors) == len(y_tensors)
            self.y_tensors = y_tensors
            self.has_y = True

        self.x_tensors = x_tensors

        self.transform = transform

    def __getitem__(self, index):
        x = self.x_tensors[index]

        if self.transform:
            x = self.transform(x)

        if self.has_y:
            y = self.y_tensors[index]

            return x, y
        else:
            return x

    def __len__(self):
        return len(self.x_tensors)

def get_cifar10_c_loader(datafolder, filename, preprocess):
    images = np.load(f'{datafolder}/{filename}.npy')

    tensor_ds = CustomTensorDataset(images, transform=preprocess)
    testLoader = torch.utils.data.DataLoader(tensor_ds, batch_size=100,
                        pin_memory=True, shuffle=False)
    return testLoader


def get_cifar10c_img_features(clip_model, dataLoader, prompt_encodings, device='cuda:0'):
    X = []
    img_features = []
    for i, x in tqdm(enumerate(dataLoader)):
        with torch.no_grad():
            clip_output = clip_forward(clip_model, x.to(device), prompt_encodings)
            X.append(clip_output[0].cpu())
            img_features.append(clip_output[2].cpu())
    return torch.cat(X), torch.cat(img_features)

def load_preprocessed_data(folder, split='train'):
    X = torch.load(os.path.join(folder, f'{split}_X_prompts.pt'))
    Y = torch.load(os.path.join(folder, f'{split}_Y.pt'))
    img_features = torch.load(os.path.join(folder, f'{split}_img_features.pt'))
    return X, Y, img_features

def zeroshot_classifier(model, classnames, templates, device='cuda:0'):
    with torch.no_grad():
        non_avg_weight_dict = {}
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            non_avg_weight_dict[classname] = class_embeddings
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        logit_scale = model.logit_scale
        zeroshot_weights *= logit_scale.exp()
    return zeroshot_weights, non_avg_weight_dict


def standarize(X):
    means = X.mean(dim=0, keepdim=True)
    stds = X.std(dim=0, keepdim=True)
    normalized_data = (X - means) / stds
    return normalized_data


def run_glm(params, train_X, train_Y, test_X, test_Y, num_classes=10, eval_percent=0.2, seed=0, device='cuda'):
    X_train, X_eval, y_train, y_eval = train_test_split(
        train_X, train_Y, test_size=0.2, random_state=seed)

    return run_glm_with_val(params, X_train, y_train, X_eval, y_eval, test_X, test_Y, num_classes=10, eval_percent=0.2, seed=0, device='cuda')

def run_glm_with_val(params, X_train, y_train, X_eval, y_eval, test_X, test_Y, num_classes=10, eval_percent=0.2, seed=0, device='cuda'):
    alpha = params['alpha']
    epsilon = params['epsilon']

    indexed_train_ds = IndexedTensorDataset(X_train, y_train)
    indexed_val_ds = torch.utils.data.TensorDataset(X_eval, y_eval)
    test_ds = torch.utils.data.TensorDataset(test_X, test_Y)

    indexed_train_loader = torch.utils.data.DataLoader(indexed_train_ds, batch_size=500, shuffle=True)
    val_loader = torch.utils.data.DataLoader(indexed_val_ds, batch_size=500, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, shuffle=False)

    linear = nn.Linear(X_train.shape[1],num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE=0.1
    NITERS=500

    # Solve the GLM path
    # output = pickle.load(open('/home/zhilif/VLM/BLIP/glm_saga_dump/imagenet-v2/alltinyIN.pkl', 'rb'))
    output = glm_saga(linear, indexed_train_loader, STEP_SIZE, NITERS, params['alpha'], val_loader=val_loader, test_loader=test_loader, epsilon=params['epsilon'], k=params['k'], do_zero=False)
    return output, linear

def remap_v2_classes(v2_test_Y):
    seq1 = np.arange(0, 1000, 1)
    seq1_sr =[str(i) for i in seq1]
    seq1_sr = sorted(seq1_sr)
    seq_map = {}
    for i in range(len(seq1_sr)):
        seq_map[i] = int(seq1_sr[i])
    for i in range(len(v2_test_Y)):
        v2_test_Y[i] = seq_map[v2_test_Y[i].item()]
    return v2_test_Y

def get_acc(model, test_X, test_Y, feature_mask, standard=False, class_mask=None, device='cpu'):
    model = model.to(device)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)
    use_test_X = test_X[:,feature_mask]
    if standard:
        use_test_X = standarize(test_X[:, feature_mask])
    if class_mask:
        return accuracy_score(test_Y.detach().cpu().numpy(), model(use_test_X)[:, class_mask].argmax(1).detach().cpu().numpy())
    return accuracy_score(test_Y.detach().cpu().numpy(), model(use_test_X).argmax(1).detach().cpu().numpy())


def subsample_data(train_img_features, train_Y, num_classes=10, shots=10, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    # rand_idx = torch.randperm(10)
    # rand_idx
    rand_idx = torch.randperm(len(train_Y), generator=g)
    train_X = train_img_features[rand_idx]
    train_Y = train_Y[rand_idx]

    select_idx = []
    sample_per_class = shots
    for i in range(num_classes):
        class_idx = torch.where(train_Y==i)[0].tolist()[:sample_per_class]
        select_idx.extend(class_idx)

    train_X_subset = train_X[select_idx]
    train_Y_subset = train_Y[select_idx]
    return train_X_subset, train_Y_subset


def get_interpolate_acc(zeroshot_weights, model_weight, test_X, test_Y, num_alpha=21, num_class=1000, device='cpu'):
    accs = []
    alphas = np.linspace(0, 1, num_alpha)
    for alpha in alphas:
        lin = nn.Linear(test_X.shape[1], num_class)
        lin.bias.data.zero_()
        lin.weight.data = alpha * model_weight.to(device) + (1-alpha)*zeroshot_weights.to(device)
        accs.append(accuracy_score(test_Y.detach().to(device).numpy(), lin(test_X.to(device)).argmax(1).detach().to(device).numpy()))
    return accs

def get_classification_mask(task):
    imagenet_folder = '/project_data/datasets/ILSVRC2012/train'
    all_classes = sorted(os.listdir(imagenet_folder))
    task_folder = None
    if task == 'r':
        task_folder = '/project_data/datasets/imagenet-r'
    if task == 'a':
        task_folder = '/project_data/datasets/imagenet-a'
    target_classes = set(sorted(os.listdir(task_folder)))
    mask = []
    for i,c in enumerate(all_classes):
        if c in target_classes:
            mask.append(i)
    return sorted(mask)

def concat_reshape_prompt(nested_feature_text, prompt_encodings, zeroshot_weights):
    assert prompt_encodings.shape[1] == zeroshot_weights.shape[1]
    assert len(nested_feature_text) == zeroshot_weights.shape[0]
    
    concat_prompts = torch.zeros((zeroshot_weights.shape[0] + prompt_encodings.shape[0], zeroshot_weights.shape[1]))
    
    concat_ptr = 0 
    prompt_ptr = 0
    for i, feature_text in tqdm(enumerate(nested_feature_text)):
        num_curr = len(feature_text)
        concat_prompts[concat_ptr:concat_ptr+num_curr] = prompt_encodings[prompt_ptr:prompt_ptr+num_curr]
        prompt_ptr+=num_curr 
        concat_prompts[concat_ptr+num_curr] = zeroshot_weights[i]
        concat_ptr+=(num_curr+1)
    return concat_prompts