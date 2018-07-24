#coding:utf-8
import cv2
import numpy as np
import os.path
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented
from .text_connect_cfg import Config as TextLineCfg
from .text_connect_cfg import cfg
class Drawbox:
    def __init__(self):
        self.mode= cfg.DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector=TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector=TextProposalConnectorOriented()
    def draw_boxes(self,image_name,boxes,scale,scores):
        img=cv2.imread(image_name)
        size=[]
        size.append(img.shape[0])
        size.append(img.shape[1])
        #keep_inds=self.filter_boxes(text_recs)
        text_recs=self.text_proposal_connector.get_text_lines(boxes,scores, size)
        for box in text_recs:      
            cv2.rectangle(img,(int(box[0]),int(box[1])), (int(box[2]),int(box[3])),(255,0,0),3)      
        cv2.imwrite("1.jpg",img)
