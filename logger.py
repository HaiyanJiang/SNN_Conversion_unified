from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
# import os
# import sys

# ## A simple torch style logger
# ## Imported from https://github.com/qymeng94/DSR

# __all__ = ['Logger', 'LoggerMonitor', 'savefig']


def savefig(fig, fname, dpi=None):
    """
    fig = logger.plot()
    savefig(fig, 'test_logger.pdf')
    fig.savefig('test_logger_3.pdf', dpi=300)
    """
    dpi = 300 if dpi == None else dpi
    fig.savefig(fname, dpi=dpi)
    plt.close()


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []
                
                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()
    
    def set_formats(self, formats):
        if self.resume: 
            pass
        assert len(self.names) == len(formats), 'Formats do not match names'
        self.formats = formats

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, (fmat, num) in enumerate(zip(self.formats, numbers)):
            self.file.write(fmat.format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        fig = plt.figure(facecolor="w", figsize=(10, 8))
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)
        return fig
    
    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


class StaticLogger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, title=None): 
        self.title = '' if title == None else title

    def set_names(self, names):
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.numbers[name] = []

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.numbers[self.names[index]].append(num)


if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])
    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.close()
    # fig = logger.plot()
    
    
    # # Example: logger monitor
    # paths = {
    #     'resadvnet20':'./code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt', 
    #     'resadvnet32':'./code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    #     'resadvnet44':'./code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt'
    #     }
    
    
    # field = ['Valid Acc.']
    # monitor = LoggerMonitor(paths)
    # monitor.plot(names=field)
    # savefig('test.eps')
    
    print(666)
    
    
    
    
