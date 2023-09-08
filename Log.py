# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''
import logging
import getpass
import sys
import torch 
import os.path as osp

class MyLog(object):
    def __init__(self, init_file=None, model_name=None):
        user=getpass.getuser()
        self.logger=logging.getLogger(user)
        self.logger.setLevel(logging.DEBUG)
        if init_file==None:
            logFile=sys.argv[0][0:-3]+'.log'
        else:
            logFile=init_file
        formatter=logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')

        logHand=logging.FileHandler(logFile,encoding="utf8")
        logHand.setFormatter(formatter)
        logHand.setLevel(logging.INFO)#只记录错误

        logHandSt=logging.StreamHandler()
        logHandSt.setFormatter(formatter)

        self.logger.addHandler(logHand)
        self.logger.addHandler(logHandSt)

        best_model_name = 'best_model.pth' if model_name is None else model_name

        self.best_model_path = osp.join(
            '/'.join(init_file.split('/')[:-1]), best_model_name)

    def debug(self,msg):
        self.logger.debug(msg)
    def info(self,msg):
        self.logger.info(msg)
    def warn(self,msg):
        self.logger.warning(msg)
    def error(self,msg):
        self.logger.error(msg)
    def critical(self,msg):
        self.logger.critical(msg)
    def save_model(self, model):
        torch.save(model.state_dict(), self.best_model_path)