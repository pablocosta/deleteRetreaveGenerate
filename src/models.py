"""Sequence to Sequence models."""
import glob
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.decoders as decoders
import src.encoders as encoders

from src.cuda import CUDA


def get_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attemptLoadModel(model, checkpointDir=None, checkpointPath=None):
    assert checkpointDir or checkpointPath

    if checkpointDir:
        epoch, checkpoint_path = get_latest_ckpt(checkpointDir)
    else:
        epoch = int(checkpointPath.split('.')[-2])

    if checkpointPath:
        model.load_state_dict(torch.load(checkpointPath))
        print('Load from %s sucessful!' % checkpointPath)
        return model, epoch + 1
    else:
        return model, 0


class SeqModel(nn.Module):
    def __init__(
        self,
        srcVocabSize,
        tgtVocabSize,
        padIdSrc,
        padIdTgt,
        batchSize,
        config=None,
    ):
        """Initialize model."""
        super(SeqModel, self).__init__()
        self.srcVocabSize = srcVocabSize
        self.tgtVocabSize = tgtVocabSize
        self.padIdSrc     = padIdSrc
        self.padIdTgt     = padIdTgt
        self.batchSize    = batchSize
        self.config       = config
        self.options      = config['model']
        self.modelType    = config['model']['model_type']

        self.srcEmbedding = nn.Embedding(
            self.srcVocabSize,
            self.options['emb_dim'],
            self.padIdSrc)
        
        if self.config['data']['share_vocab']:
            self.tgtEmbedding = self.srcEmbedding
        else:
            self.tgtEmbedding = nn.Embedding(
                self.tgtVocabSize,
                self.options['emb_dim'],
                self.padIdTgt)
        
        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'])
            self.ctxBridge = nn.Linear(
                self.options['src_hidden_dim'],
                self.options['tgt_hidden_dim'])

        else:
            raise NotImplementedError('unknown encoder type')
        
        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        
        if self.modelType == 'delete':
            self.attributeEmbedding = nn.Embedding(
                num_embeddings=2, 
                embedding_dim=self.options['emb_dim'])
            attrSize = self.options['emb_dim']
        
        elif self.modelType == 'delete_retrieve':
            self.attributeEncoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'],
                pack=False)
            attrSize = self.options['src_hidden_dim']

        elif self.modelType == 'seq2seq':
            attrSize = 0
        else:
            raise NotImplementedError('unknown model type: %s. Accepted values: [seq2seq, delete_retrieve, delete]' % self.model_type)
        
        self.cBridge = nn.Linear(
            attrSize + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])
        self.hBridge = nn.Linear(
            attrSize + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])

        # # # # # #  # # # # # #  # # # # # END NEW STUFF
        self.decoder = decoders.StackedAttentionLSTM(config=config)

        self.outputProjection = nn.Linear(
            self.options['tgt_hidden_dim'],
            tgtVocabSize)

        self.softmax = nn.Softmax(dim=-1)

        self.initWeights()

    def initWeights(self):
        """Initialize weights."""
        initRange = 0.1
        self.srcEmbedding.weight.data.uniform_(-initRange, initRange)
        self.tgtEmbedding.weight.data.uniform_(-initRange, initRange)
        self.cBridge.bias.data.fill_(0)
        self.outputProjection.bias.data.fill_(0)

    def forward(self, inputSrc, inputTgt, srcMask, srcLens, inputAttr, attrLens, attrMask):
        
        srcEmb                     = self.srcEmbedding(inputSrc)
        srcmask                    = (1-srcMask).byte()
        srcOutputs, (srcHT, srcCT) = self.encoder(srcEmb, srcLens, srcMask)

        #bidirectional
        if self.options['bidirectional']:
            hT = torch.cat((srcHT[-1], srcHT[-2]), 1)
            cT = torch.cat((srcCT[-1], srcCT[-2]), 1)
        else:
            hT = srcHT[-1]
            cT = srcCT[-1]

        srcOutputs = self.ctxBridge(srcOutputs)

        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        if self.modelType == 'delete':
            # just do h i guess?
            aHt = self.attributeEmbedding(inputAttr)
            cT = torch.cat((cT, aHt), -1)
            hT = torch.cat((hT, aHt), -1)

        elif self.modelType == 'delete_retrieve':
            attrEmb = self.srcEmbedding(inputAttr)
            _, (aHt, aCt) = self.attributeEncoder(attrEmb, attrLens, attrMask)
            if self.options['bidirectional']:
                aHt = torch.cat((aHt[-1], aHt[-2]), 1)
                aCt = torch.cat((aCt[-1], aCt[-2]), 1)
            else:
                aHt = aHt[-1]
                aCt = aCt[-1]

            hT = torch.cat((hT, aHt), -1)
            cT = torch.cat((cT, aCt), -1)
            
        cT = self.cBridge(cT)
        hT = self.hBridge(hT)

        # # # #  # # # #  # #  # # # # # # #  # # end diff
        
        tgtEmb = self.tgtEmbedding(inputTgt)
        tgtOutputs, (_, _) = self.decoder(
            tgtEmb,
            (hT, cT),
            srcOutputs,
            srcMask)

        tgtOutputsReshape = tgtOutputs.contiguous().view(
            tgtOutputs.size()[0] * tgtOutputs.size()[1],
            tgtOutputs.size()[2])
        decoderLogit = self.outputProjection(tgtOutputsReshape)
        decoderLogit = decoderLogit.view(
            tgtOutputs.size()[0],
            tgtOutputs.size()[1],
            decoderLogit.size()[1])

        probs = self.softmax(decoderLogit)

        return decoderLogit, probs

    def countParams(self):
        nParams = 0
        for param in self.parameters():
            nParams += np.prod(param.data.cpu().numpy().shape)
        return nParams