# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import logging
    
logger = logging.getLogger(__name__)    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.bert = RobertaForSequenceClassification.from_pretrained("neulab/codebert-cpp")
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

    
        
    def forward(self, input_ids=None,labels=None): 
        logger.info("Hailong: the input_ids : " + str(input_ids))
        logger.info(f"Hailong: the shape of input_ids is : " +str(input_ids.size()) + "\n")
        logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logger.info("Hailong: the logits : " + str(logits))
        #prob=torch.softmax(logits,-1)
        prob = logits
        logger.info("Hailong: the probs : " + str(prob))
        #logger.info( prob) 
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits,labels)
            logger.info("Hailong: the loss is : " + str(loss)) 
            return loss,prob
        else:
            return prob
      
        
 
