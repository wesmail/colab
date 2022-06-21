'''
Author: Wojciech Fedorko
Collaborators: Julian Ding, Abhishek Kajal
'''

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim.lr_scheduler import StepLR

import os
import sys
import time
import numpy as np

from typing import Optional
from operator import itemgetter

class CSVData:

    def __init__(self,fout):
        self.name  = fout
        self._fout = None
        self._str  = None
        self._dict = {}

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):
        if self._str is None:
            self._fout=open(self.name,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()
    
    def close(self):
        if self._str is not None:
            self._fout.close()


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper for making general samplers compatible with multiprocessing.
    Allows you to use any sampler in distributed mode when training with
    torch.nn.parallel.DistributedDataParallel. In such case, each process
    can pass a DistributedSamplerWrapper instance as a DataLoader sampler,
    and load a subset of subsampled data of the original dataset that is
    exclusive to it.
    """

    def __init__(
        self,
        sampler,
        seed,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Args:
            sampler                         ... Sampler used for subsampling
            num_replicas (int, optional)    ... Number of processes participating in distributed training
            rank (int, optional)            ... Rank of the current process within ``num_replicas``
            shuffle (bool, optional)        ... If true sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            list(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        self.sampler = sampler
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # fetch DistributedSampler indices
        indexes_of_indexes = super().__iter__()

        # deterministically shuffle based on epoch
        updated_seed = self.seed + int(self.epoch)
        torch.manual_seed(updated_seed)

        # fetch subsampler indices with synchronized seeding
        subsampler_indices = list(self.sampler)

        # get subsampler_indexes[indexes_of_indexes]
        distributed_subsampler_indices = itemgetter(*indexes_of_indexes)(subsampler_indices)

        return iter(distributed_subsampler_indices)

class Engine:
    """The training engine

    Performs training and evaluation
    """

    def __init__(self, rank, model, gpu, train_dataset, config, is_graph=False):
        self.model = model
        self.device = torch.device(gpu)
        self.rank = rank
        self.world_size = config.world_size
        self.is_graph = is_graph

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)

        # gamma = decaying factor
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None


        self.train_dataset = train_dataset

        self.indices = np.load(config.indices)
        self.seed = np.random.randint(100000)

        train_sampler = SubsetRandomSampler(self.indices['train_idxs'])
        val_sampler = SubsetRandomSampler(self.indices['val_idxs'])
        test_sampler = SubsetRandomSampler(self.indices['test_idxs'])
        #train_sampler = DistributedSamplerWrapper(sampler=train_sampler, rank=self.rank, seed=self.seed)
        #val_sampler = DistributedSamplerWrapper(sampler=val_sampler, rank=self.rank, seed=self.seed)
        #test_sampler = DistributedSamplerWrapper(sampler=test_sampler, rank=self.rank, seed=self.seed)

        self.train_dldr = None
        self.val_dldr   = None
        self.test_dldr  = None

        if not self.is_graph:
            self.train_dldr=DataLoader(self.train_dataset,
                                       batch_size=config.batch_size_train,
                                       shuffle=False,
                                       sampler=train_sampler,
                                       num_workers=0)
            self.val_dldr=DataLoader(self.train_dataset,
                                     batch_size=config.batch_size_val,
                                     shuffle=False,
                                     sampler=val_sampler,
                                     num_workers=0)
            self.test_dldr=DataLoader(self.train_dataset,
                                      batch_size=config.batch_size_test,
                                      shuffle=False,
                                      sampler=val_sampler,
                                      num_workers=0)
        else:
            self.train_dldr=PyGDataLoader(self.train_dataset, \
                                       batch_size=config.batch_size_train, \
                                       shuffle=False, \
                                       sampler=train_sampler, \
                                       num_workers=0)
            self.val_dldr=PyGDataLoader(self.train_dataset, \
                                     batch_size=config.batch_size_val, \
                                     shuffle=False, \
                                     sampler=val_sampler, \
                                     num_workers=0)
            self.test_dldr=PyGDataLoader(self.train_dataset, \
                                      batch_size=config.batch_size_test, \
                                      shuffle=False, \
                                      sampler=test_sampler, \
                                      num_workers=0)

        self.val_iter=iter(self.val_dldr)
        self.test_iter=iter(self.test_dldr)

        self.dirpath=config.dump_path + "/"+time.strftime("%Y%m%d_%H%M%S") + "/"


        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump: {}".format(self.dirpath))
            os.makedirs(self.dirpath,exist_ok=True)

        self.config=config

        # Save a copy of the config in the dump path
        f_config=open(self.dirpath+"/config_log.txt","w")
        f_config.write(str(vars(config)))


    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU
            # if using CPU this has no effect
            self.data = self.data.to(self.device)
            self.label = self.label.to(self.device)


            linear_model_out = self.model(self.data)
            # Training

            self.loss = self.criterion(linear_model_out,self.label)


            softmax    = self.softmax(linear_model_out).detach().cpu().numpy()
            prediction = torch.argmax(linear_model_out,dim=-1)
            accuracy   = (prediction == self.label).sum().item() / float(prediction.nelement())
            prediction = prediction.cpu().numpy()

        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : self.loss.detach().cpu().item(),
                'accuracy'   : accuracy}

    def backward(self):
        self.optimizer.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.optimizer.step()

    # ========================================================================
    def train(self, epochs=3.0, report_interval=10, valid_interval=1000):
        # Based on WaTCHMaL workshop and W's code

        # Keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss=1.0e6

        # Prepare attributes for data logging
        self.train_log, self.val_log = CSVData(self.dirpath+"log_train.csv"), CSVData(self.dirpath+"log_val.csv")
        # Set neural net to training mode
        self.model.train()
        # Initialize epoch counter
        epoch = 0.
        # Initialize iteration counter
        self.iteration = 0
        # Training loop
        while ((int(epoch+0.5) < epochs) ):
            if self.rank == 0:
                print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            j = 0
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_dldr):

                # once in a while run valiation
                # as a sanity check run validation before we start training
                if i%valid_interval == 0:
                    self.model.eval()

                    try:
                        val_data = next(self.val_iter)
                    except StopIteration:
                        print("starting over on the validation set")
                        self.val_iter=iter(self.val_dldr)
                        val_data = next(self.val_iter)

                    # Data and label
                    if not self.is_graph:
                        self.data = val_data[0]
                        self.label = val_data[1]
                    else:
                        self.data = val_data
                        self.label = val_data.y

                    res = self.forward(False)
                    if self.rank == 0:
                        print('... Iteration %d ... Epoch %1.2f ... Validation Loss %1.3f ... Validation Accuracy %1.3f' % (self.iteration,epoch,res['loss'],res['accuracy']))


                    self.model.train()

                    self.save_state()
                    mark_best=0
                    if res['loss']<best_val_loss:
                        best_val_loss=res['loss']
                        if self.rank == 0:
                            print('best validation loss so far!: {}'.format(best_val_loss))
                        self.save_state(best=True)
                        mark_best=1

                    self.val_log.record(['iteration','epoch','accuracy','loss','saved_best'],[self.iteration,epoch,res['accuracy'],res['loss'],mark_best])
                    self.val_log.write()
                    self.val_log.flush()

                # Data and label
                if not self.is_graph:
                    self.data = data[0]
                    self.label = data[1]
                else:
                    self.data = data
                    self.label = data.y


                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                # Call backward: backpropagate error and update weights
                self.backward()
                # Epoch update
                epoch += 1./len(self.train_dldr)
                self.iteration += 1

                # Log/Report
                #
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','accuracy','loss'],[self.iteration,epoch,res['accuracy'],res['loss']])
                self.train_log.write()
                self.train_log.flush()

                # once in a while, report
                if i==0 or i%report_interval == 0 and self.rank == 0:
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (self.iteration,epoch,res['loss'],res['accuracy']))
                    pass



                if epoch >= epochs:
                    break

        # Decay Learning Rate
        self.scheduler.step()
        self.val_log.close()
        self.train_log.close()
        #np.save(self.dirpath + "/optim_state_array.npy", np.array(optim_state_list))

    # ========================================================================

    # Function to test the model performance on the validation
    # dataset ( returns loss, acc, confusion matrix )
    def validate(self, plt_worst=0, plt_best=0):
        """
        Test the trained model on the validation set.

        Parameters: None

        Outputs :
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy

        Returns : None
        """


        # Variables to output at the end
        val_loss = 0.0
        val_acc = 0.0
        val_iterations = 0

        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():

            # Set the model to evaluation mode
            self.model.eval()

            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes= [],[],[],[],[]

            # Extract the event data and label from the DataLoader iterator
            for it, val_data in enumerate(self.val_dldr):

                sys.stdout.write("val_iterations : " + str(val_iterations) + "\n")

                # Data and label
                if not self.is_graph:
                    self.data = val_data[0]
                    self.label = val_data[1]
                else:
                    self.data = val_data
                    self.label = val_data.y



                # Run the forward procedure and output the result
                result = self.forward(False)
                val_loss += result['loss']
                val_acc += result['accuracy']

                # Add item to priority queues if necessary

                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")

                # Add the local result to the final result
                labels.extend(self.label)
                predictions.extend(result['prediction'])
                softmaxes.extend(result['softmax'])

                val_iterations += 1

        print(val_iterations)

        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)

        np.save(self.dirpath + "val_labels.npy", np.array(labels))
        np.save(self.dirpath + "val_predictions.npy", np.array(predictions))
        np.save(self.dirpath + "val_softmax.npy", np.array(softmaxes))

    # Function to test the model performance on the test
    # dataset ( returns loss, acc, confusion matrix )
    def test(self, plt_worst=0, plt_best=0):
        """
        Test the trained model on the test set.

        Parameters: None

        Outputs :
            total_test_loss = accumulated test loss
            avg_test_loss = average test loss
            total_test_acc = accumulated test accuracy
            avg_test_acc = accumulated test accuracy

        Returns : None
        """


        # Variables to output at the end
        test_loss = 0.0
        test_acc = 0.0
        test_iterations = 0

        # Iterate over the test set to calculate val_loss and val_acc
        with torch.no_grad():

            # Set the model to evaluation mode
            self.model.eval()

            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes= [],[],[],[],[]

            # Extract the event data and label from the DataLoader iterator
            for it, test_data in enumerate(self.test_dldr):

                sys.stdout.write("test_iterations : " + str(test_iterations) + "\n")

                # Data and label
                if not self.is_graph:
                    self.data = test_data[0]
                    self.label = test_data[1]
                else:
                    self.data = test_data
                    self.label = test_data.y



                # Run the forward procedure and output the result
                result = self.forward(False)
                test_loss += result['loss']
                test_acc += result['accuracy']

                # Add item to priority queues if necessary

                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")

                # Add the local result to the final result
                labels.extend(self.label)
                predictions.extend(result['prediction'])
                softmaxes.extend(result['softmax'])

                test_iterations += 1

        print(test_iterations)

        print("\nTotal test loss : ", test_loss,
              "\nTotal test acc : ", test_acc,
              "\nAvg test loss : ", test_loss/test_iterations,
              "\nAvg test acc : ", test_acc/test_iterations)

        np.save(self.dirpath + "test_labels.npy", np.array(labels))
        np.save(self.dirpath + "test_predictions.npy", np.array(predictions))
        np.save(self.dirpath + "test_softmax.npy", np.array(softmaxes))


    # ========================================================================


    def save_state(self,best=False):
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST" if best else ""),
                                     ".pth")
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        if self.rank == 0:
            print('Saved checkpoint as:', filename)
        return filename

    def restore_state(self, weight_file):

        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
        print('Restoration complete.')
