# Newest 2018.11.23 9:53:00

from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import PIL
import copy

from PIL import Image
#from keras.utils import to_categorical
import pickle
import config as cf
import numpy as np

import os
import sys
import time
import argparse
import matplotlib.pyplot as plt

from dataset import get_all_dataLoders, get_dataLoder
from models import Expert_gate, expert, autoencoder, Clustering
from losses import pred_loss, consistency_loss, recon_loss, cross_entropy
from utils import get_optim, get_all_assign, stack_or_create


parser = argparse.ArgumentParser(description='HD_CNN in PyTorch')
parser.add_argument('--dataset', default='cifar-100', type=str, help='Determine which dataset to be used')
parser.add_argument('--num_superclasses', default=2, type=int, help='The number of cluster centers')
#parser.add_argument('--num_epochs_pretrain', default=1, type=int, help='The number of pre-train epoches')
parser.add_argument('--num_epochs_train', default=50, type=int, help='The number of train epoches')
parser.add_argument('--pretrain_batch_size', default=32, type=int, help='The batch size of pretrain')
parser.add_argument('--train_batch_size', default=32, type=int, help='The batch size of train')
parser.add_argument('--min_classes', default=1, type=int, help='The minimum of classes in one superclass')
parser.add_argument('--num_test', default= 5000, type=int, help='The number of test batch steps in a epoch ================= when final test, make it bigger')
parser.add_argument('--rel_th', default=0.75 , type=float, help='Determine the threshold')

parser.add_argument('--opt_type', default='adam', type=str, help='Determine the type of the optimizer')
parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='The learning rate of pre-training')
parser.add_argument('--finetune_lr', default=0.001, type=float, help='The learning rate of inner model')
parser.add_argument('--drop_rate', default=0.5, type=float, help='The probability of to keep')
parser.add_argument('--weight_consistency', default=1e1, type=float, help='The weight of coarse category consistency')
parser.add_argument('--gamma', default=5, type=float, help='The weight for u_k')#1.25

parser.add_argument('--resume_model', default= True, type=bool, help='resume the whole model from checkpoint')
args = parser.parse_args()

# Hyper Parameter settings
cf.use_cuda = torch.cuda.is_available()

trainloader, testloader, validloader, encoded_trainloader, encoded_testloader, encoded_validloader = get_all_dataLoders(args, valid=True, one_hot=True)
args.num_classes = 10 if args.dataset == 'cifar-10' else 100

# Data class Preparation
Class_generator = {}
for i in range(100):
    Class_generator[i]= i


# Model
print('\nModel setup')
net = Expert_gate(args)
function = Clustering(args)
Expert = {}
Autoencoder = {}
Superclass = {}
Old_superclass = {}


def save_and_load(Superclass, Autoencoder, Old_superclass, num = 0 ,load = False):
    save_point = cf.var_dir + args.dataset
    if not load:
        if (num+1)%10 == 0 and (num+1) != 0:
            torch.save(Superclass ,save_point+'/classes'+str(num)+'_Superclass.pkl')
            torch.save(Old_superclass ,save_point+'/classes'+str(num)+'_old_Superclass.pkl')
            torch.save(Autoencoder ,save_point+'/classes'+str(num)+'_Autoencoder.pkl')
            print('\n================= (num%10==0) Model saving finish ================')
        print('Save the Autoencoder with keys:',Autoencoder.keys())
        print('Save the superclass:',Superclass)
        print('Save the Old_superclass:',Old_superclass)
        torch.save(Superclass ,save_point+'/Superclass.pkl')
        torch.save(Old_superclass ,save_point+'/old_Superclass.pkl')
        torch.save(Autoencoder ,save_point+'/Autoencoder.pkl')
        print('\n================= Model saving finish ================')
    else:
        if os.path.exists(save_point+'/Superclass.pkl'):
            Superclass = torch.load(save_point+'/Superclass.pkl')
            print('\n Resume the whole superclass', Superclass)
        if os.path.exists(save_point+'/old_Superclass.pkl'):
            Old_superclass = torch.load(save_point+'/old_Superclass.pkl')
            print('\n Resume the old_superclass', Old_superclass)
        if os.path.exists(save_point+'/Autoencoder.pkl'):
            Autoencoder = torch.load(save_point+'/Autoencoder.pkl')
            print('\n Resume the whole Autoencoders', Autoencoder.keys())
        if os.path.exists(save_point+'/Expert.pkl'):
            Expert = torch.load(save_point+'/Expert.pkl')
            print('\n Resume the whole expert model', Expert.keys())
        else:
            Expert = {}
        return Superclass, Autoencoder, Expert, Old_superclass

# ============= used to train the new encoder and classify it ===========
def train_autoencoder(Autoencoder, Superclass, Old_superclass):
    # ================== used to train the new encoder ==================
    print('\n=========== refesh the autoencoders ===========')
    for dict in Superclass:
        refresh = 'false'
        if dict not in Old_superclass.keys():
            refresh = 'true'
        elif Superclass[dict]!= Old_superclass[dict]:
            refresh = 'true'
        if refresh=='true':
            print('\nrefeshing the autoencoder:'+dict)
            Autoencoder[dict] = autoencoder(args)
            if cf.use_cuda:
                Autoencoder[dict].cuda()
                cudnn.benchmark = True
            for epoch in range(args.num_epochs_train):
                Autoencoder[dict].train()
                required_train_loader = get_dataLoder(args, classes = Superclass[dict], mode='Train',encoded=True, one_hot=False)
                param = list(Autoencoder[dict].parameters())
                optimizer, lr = get_optim(param, args, mode='preTrain', epoch=epoch)
                for batch_idx, (inputs, targets) in enumerate(required_train_loader):
                    if batch_idx>=args.num_test:
                        break
                    if cf.use_cuda:
                        inputs = inputs.cuda() # GPU settings
                    optimizer.zero_grad()
                    inputs = Variable(inputs)
                    reconstructions, _ = Autoencoder[dict](inputs)
                    loss = cross_entropy(reconstructions, inputs)
                    loss.backward()  # Backward Propagation
                    optimizer.step() # Optimizer update
                    sys.stdout.write('\r')
                    sys.stdout.write('Refreshing autoencoder:' + dict + ' with Epoch [%3d/%3d] Iter [%3d]\t\t Loss: %.4f'
                                     %(epoch+1, args.num_epochs_train, batch_idx+1, loss.item()))
                    sys.stdout.flush()
            print('\nautoencoder model:'+str(dict)+' is constrcuted with final loss:'+str(loss.item()))
    return Autoencoder


# ============= used to train the new encoder and classify it ===========
def train_test_autoencoder(newclasses, Autoencoder):
    # ================== used to train the new encoder ==================
    Autoencoder[str(newclasses)] = autoencoder(args)
    if cf.use_cuda:
        Autoencoder[str(newclasses)].cuda()
        cudnn.benchmark = True
    for epoch in range(args.num_epochs_train):
        Autoencoder[str(newclasses)].train()
        required_train_loader = get_dataLoder(args, classes = [newclasses], mode='Train',encoded=True, one_hot=True)
        param = list(Autoencoder[str(newclasses)].parameters())
        optimizer, lr = get_optim(param, args, mode='preTrain', epoch=epoch)
        print('\n==> Epoch #%d, LR=%.4f' % (epoch+1, lr))
        for batch_idx, (inputs, targets) in enumerate(required_train_loader):
            if batch_idx>=args.num_test:
                break
            if cf.use_cuda:
                inputs = inputs.cuda() # GPU settings
            optimizer.zero_grad()
            inputs = Variable(inputs)
            reconstructions, _ = Autoencoder[str(newclasses)](inputs)
            loss = cross_entropy(reconstructions, inputs)
            loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update
            sys.stdout.write('\r')
            sys.stdout.write('Train autoencoder:' + str(newclasses) + ' with Epoch [%3d/%3d] Iter [%3d]\t\t Loss: %.4f'
                             %(epoch+1, args.num_epochs_train, batch_idx+1, loss.item()))
            sys.stdout.flush()
    # =============== used to classify it and nut it in a proper superclass ==============
    if Autoencoder:
        Loss={}
        Rel={}
        print('\ntesting the new data in previous autoencoders')
        for dict in Autoencoder:
            Loss[dict]=0
            required_valid_loader = get_dataLoder(args, classes = [int(dict)], mode='Valid',encoded=True, one_hot=True)
            for batch_idx, (inputs, targets) in enumerate(required_valid_loader):
                if batch_idx >= args.num_test:
                    break
                if cf.use_cuda:
                    inputs = inputs.cuda() # GPU settings
                inputs = Variable(inputs)
                reconstructions, _ = Autoencoder[dict](inputs)
                loss = cross_entropy(reconstructions, inputs)
                Loss[dict] += loss.data.cpu().numpy() if cf.use_cuda else loss.data.numpy()
        print('\nAutoencoder:'+str(newclasses)+' is been delated and wait for update for every ten classes')
        Autoencoder.pop(str(newclasses),'\nthe class:'+str(newclasses)+' is not been delated as the dict not exist')
        highest = 0
        test_result=''
        for dict in Loss:
            Rel[dict]=1-abs((Loss[dict]-Loss[str(newclasses)])/Loss[str(newclasses)])
            if Rel[dict] >= highest and Rel[dict]>=args.rel_th and dict!=str(newclasses):
                highest = Rel[dict]
                test_result = dict
                print('\nnewclass:'+str(newclasses)+' is add to superclass with class:' + dict)
        print('\nClass rel:', Rel, ' and Loss:', Loss)
        return Autoencoder , test_result
    else:
        return Autoencoder , _


def save_model(model, dict, name = 'Expert'):
    print('\nSaving the expert model dict: Expert'+dict)
    save_point = cf.var_dir + args.dataset
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    share_variables = model.state_dict()
    if name == 'Expert':
        share_variables.pop('fc2.bias','dict: fc2.bias dot found in dictionary share_variables')
        share_variables.pop('fc2.weight','dict: fc2.weight dot found in dictionary share_variables')
    print('\nSaving the full expert model: Expert'+dict)
    torch.save(model, save_point + '/full_Expert'+ dict+'.pkl')
    torch.save(share_variables, save_point + '/'+ name + dict+'.pkl')

def prepared_model(num_output, dict):
    save_point = cf.var_dir + args.dataset
    model = expert(args, num_output)
    if len(dict)>2 and os.path.exists(save_point + '/Expert'+dict+'.pkl'):
        print('\nLoad the model Expert'+dict)
        variables = torch.load(save_point + '/Expert'+dict+'.pkl')
        needs = {}
        for need in model.state_dict():
            if need in variables:
                needs[need] = variables[need]
            else:
                needs[need] = model.state_dict()[need]
        model.load_state_dict(needs)
    return model


# =============== just trian the expert model after generating the newsuperclass ==============
def enhance_expert(Expert, Superclass, c, mode = 'clone'):
    if mode == 'clone':
        print('\nThe new expert model is activate and waiting for another class added to build')
    elif mode=='merge':
        for epoch in range(args.num_epochs_train):
            required_train_loader = get_dataLoder(args, classes= Superclass[c], mode='Train', encoded=False, one_hot=True)
            if epoch == 0:
                num = len(Superclass[c])
                Expert[c] = prepared_model(num, c)
                if cf.use_cuda:
                    Expert[c].cuda()
                    cudnn.benchmark = True
            param = list(Expert[c].parameters())
            optimizer, lr = get_optim(param, args, mode='preTrain', epoch=epoch)
            for batch_idx, (inputs, targets) in enumerate(required_train_loader):
                if batch_idx>=args.num_test:
                    break
                if cf.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda() # GPU setting
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets).long()
                outputs = Expert[c](inputs) # Forward Propagation
                loss = pred_loss(outputs,targets)
                loss.backward()  # Backward Propagation
                optimizer.step() # Optimizer update
                _, predicted = torch.max(outputs.data, 1)
                num_ins = targets.size(0)
                correct = predicted.eq((torch.max(targets.data, 1)[1])).cpu().sum()
                acc=100.*correct.item()/num_ins
                sys.stdout.write('\r')
                sys.stdout.write('Train expert model with Epoch [%3d/%3d] Iter [%3d]\t\t Loss: %.4f Accuracy: %.3f%%'
                                 %(epoch+1, args.num_epochs_train, batch_idx+1, loss.item(), acc))
                sys.stdout.flush()
        save_model(Expert[c], c)
    else:
        print('\nmode error')
    return Expert

# def clone_or_merge(Autoencoder, Expert, Superclass, newclasses):
#     if Autoencoder:
#         print('\nAutoencoder is not empty')
#         Autoencoder , result = train_test_autoencoder(newclasses, Autoencoder)
#         if result:
#             c = [k for k, v in Superclass.items() if result in v]
#             Superclass[c] += [newclasses]
#             Expert = enhance_expert(Expert, Superclass, newclasses, c, mode = 'clone')
#         else:
#             c = str(len(Superclass))
#             Superclass[c] = [newclasses]
#             Expert = enhance_expert(Expert, Superclass, newclasses, c, mode = 'merge')
#     else:
#         print('\nAutoencoder is empty')
#         Autoencoder  = train_test_autoencoder(newclasses, Autoencoder)
#         Superclass['0']=[newclasses]
#     return Autoencoder, Expert, Superclass

def load_expert(Expert):
    save_point = cf.var_dir + args.dataset
    for i in range(100):
        if os.path.exists(share_variables, save_point + '/Expert' + str(i)+'.pkl'):
            Expert[str(i)]= torch.load(share_variables, save_point + '/Expert' + str(i)+'.pkl')
    return Expert

# ============================= constrcut the model =================================
if args.resume_model:
    print('\n============== Construct the incremental learning model ==============')
    print('\n============== Resume the incremental learning model ==============')
    Superclass, Autoencoder, Expert, Old_superclass = save_and_load(Superclass, Autoencoder, Old_superclass, load=True)
    for newclasses in Class_generator:
        stay_or_not = 'True'
        for dict in Superclass:
            if newclasses in Superclass[dict]:
                stay_or_not = 'False'
                break
        if stay_or_not == 'False':
            continue
        if newclasses<9:
            Superclass[str(newclasses)]=[Class_generator[newclasses]]
            continue
        elif newclasses==9:
            Superclass[str(newclasses)]=[Class_generator[newclasses]]
            Autoencoder = train_autoencoder(Autoencoder, Superclass, Old_superclass)
            Old_superclass = copy.deepcopy(Superclass)
            print('\nrefresh the old superclass')
            continue
        else:
            print('\ntrain the model normally with added class:', newclasses)
        if Autoencoder:
            print('\nAutoencoder is not empty')
            Autoencoder , result = train_test_autoencoder(newclasses, Autoencoder) # used to train the new encoder and classify it
            if result:
                c = result
                Superclass[c] += [newclasses]
            else:
                c = str(len(Superclass))
                Superclass[c] = [newclasses]
        else:
            print('\nAutoencoder is empty')
            Autoencoder ,_ = train_test_autoencoder(newclasses, Autoencoder)
            Superclass['0']=[newclasses]
        print('\n============== new classes added to model ==============\n')
        print(Superclass)
        print('\n================= end of superclass for now ================')
        if (newclasses+1)%10 == 0:
            Autoencoder = train_autoencoder(Autoencoder, Superclass, Old_superclass)
            Old_superclass = copy.deepcopy(Superclass)
            print('\nwhen processing at class:'+str(newclasses)+' ====== refresh the old superclass')
        save_and_load(Superclass, Autoencoder, Old_superclass,  num = newclasses)
    print('\n================= Model construction finish ================')

# ============================= test the model =================================
if args.resume_model:
    print('\n============== Resume the incremental learning model ==============')
    print('\n=============== test the incremental learning model ===============')
    Superclass, Autoencoder, Expert, _ = save_and_load(Superclass, Autoencoder, Old_superclass, load=True)
    save_point = cf.var_dir + args.dataset
    for dict in Superclass:
        if dict=='0' and Expert:
            break
        if os.path.exists(save_point+'/full_Expert'+dict+'.pkl'):
            Expert[dict]= torch.load(save_point+'/full_Expert'+dict+'.pkl')
            print('\nResume model: full_Expert'+dict+' to Expert')
            continue
        if len(Superclass[dict]) > 1:
            Expert = enhance_expert(Expert, Superclass, dict, mode = 'merge')
    if not os.path.exists(save_point+'/Expert.pkl'):
        print('\nSave the whole Expert model')
        torch.save(Expert,save_point+'/Expert.pkl')
    result={}
    result['correct']=0
    result['num']=0
    for newclass in Class_generator:
        result['correct_'+str(newclass)] = 0
        result['num_'+str(newclass)] = 0
        result['root_correct_'+str(newclass)] = 0
        result['class_num'+str(newclass)] =0
        result['class_correct'+str(newclass)] =0
        required_test_loader = get_dataLoder(args, classes=[Class_generator[newclass]], mode='Test', encoded=False, with_encoded=True, one_hot= False)
        for batch_idx, (inputs, targets) in enumerate(required_test_loader) :
            if batch_idx>=args.num_test:
                break
            if cf.use_cuda:
                inputs, targets, encoded_inputs = inputs.cuda(), targets.cuda(), encoded_inputs.cuda() # GPU setting
            inputs, encoded_inputs, targets = Variable(inputs), Variable( encoded_inputs) , Variable(targets).long()
            for i in range(inputs.shape[0]):
                input = inputs[i:i+1,: ,: ,: ]
                encoded_inputs = encoded_inputs[i:i+1,: ,: ,: ]
                target = targets[i].data.cpu().numpy()
                A_out ={}
                highest = {}
                highest['rel'] = float("inf")
                highest['dict'] = ''
                for a in Autoencoder:
                    if cf.use_cuda:
                        Autoencoder[a].cuda()
                        cudnn.benchmark = True
                    try:
                        reconstruction, _ = Autoencoder[a](encoded_inputs)

                    except RuntimeError:
                        continue
                    loss = cross_entropy(reconstruction, input)
                    if loss.data.cpu().numpy() <= highest['rel']:
                        highest['rel'] = loss.data.cpu().numpy()
                        highest['dict'] = a
                s_class = [k for k, v in Superclass.items() if int(highest['dict']) in v][0]
                if target in Superclass[s_class]:
                    result['root_correct_'+str(newclass)] +=1
                if len(Superclass[s_class])>1:
                    if cf.use_cuda:
                        Expert[s_class].cuda()
                        cudnn.benchmark = True
                    output = Expert[s_class](input)
                    output = torch.argmax(output,1)
                    output = Superclass[s_class][output.data.cpu().numpy()[0]]
                    if target in Superclass[s_class]:
                        result['class_num'+str(newclass)] +=1
                else:
                    output = Superclass[s_class][0]
                if output==target:
                    if len(Superclass[s_class])>1 and target in Superclass[s_class]:
                        result['class_correct'+str(newclass)] +=1
                    result['correct_'+str(newclass)] += 1
                result['num_'+str(newclass)] += 1
                sys.stdout.write('\r')
                sys.stdout.write('Class:'+str(newclass)+' have correct classfication:'+str(result['class_correct'+str(newclass)])
                                 +'/'+str(result['class_num'+str(newclass)])+' have root correct:'+str(result['root_correct_'+str(newclass)])
                                 +' and have correct:'+str(result['correct_'+str(newclass)])+' and total:'+str(result['num_'+str(newclass)]))
                sys.stdout.flush()
        result['correct'] += result['correct_'+str(newclass)]
        result['num'] += result['num_'+str(newclass)]
        result['acc_'+str(newclass)] = result['correct_'+str(newclass)]/result['num_'+str(newclass)]
        print('\nClass:'+str(newclass)+' have acc:'+ str(100*result['acc_'+str(newclass)])+'% and have root acc:'+str(100*result['root_correct_'+str(newclass)]/result['num_'+str(newclass)])+'%')
    result['acc'] = result['correct']/result['num']
    print('\n ========== Total acc:'+ str(result['acc'])+' ==========')








