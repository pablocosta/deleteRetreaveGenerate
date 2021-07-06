import sys

import json
import numpy as np
import logging
import argparse
import os
import time
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import src.evaluation as evaluation
from src.cuda import CUDA
import src.data as data
import src.models as models



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument(
    "--bleu",
    help="do BLEU eval",
    action='store_true'
)
parser.add_argument(
    "--overfit",
    help="train continuously on one batch of data",
    action='store_true'
)
args = parser.parse_args()
config = json.load(open(args.config, 'r'))

workingDir = config['data']['working_dir']

if not os.path.exists(workingDir):
    os.makedirs(workingDir)

config_path = os.path.join(workingDir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % workingDir,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('Reading data ...')
src, tgt = data.read_nmt_data(
    src=config['data']['src'],
    config=config,
    tgt=config['data']['tgt'],
    attribute_vocab=config['data']['attribute_vocab'],
    ngram_attributes=config['data']['ngram_attributes']
)


srcTest, tgtTest = data.read_nmt_data(
    src=config['data']['src_test'],
    config=config,
    tgt=config['data']['tgt_test'],
    attribute_vocab=config['data']['attribute_vocab'],
    ngram_attributes=config['data']['ngram_attributes'],
    train_src=src,
    train_tgt=tgt
)
logging.info('...done!')

logging.info('...done!')

#model configs

batchSize    = config['data']['batch_size']
maxLength    = config['data']['max_len']
srcVocabSize = len(src['tok2id'])
tgtVocabSize = len(tgt['tok2id'])

weightMask                         = torch.ones(tgtVocabSize)
weightMask[tgt['tok2id']['<pad>']] = 0
lossCriterion                      = nn.CrossEntropyLoss(weight=weightMask)

if CUDA:
    weightMask    = weightMask.cuda()
    lossCriterion = lossCriterion.cuda()

torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])


#model definition

model = models.SeqModel(
    srcVocabSize=srcVocabSize,
    tgtVocabSize=tgtVocabSize,
    padIdSrc=src['tok2id']['<pad>'],
    padIdTgt=tgt['tok2id']['<pad>'],
    batchSize=batchSize,
    config=config
)

logging.info('MODEL HAS %s params' %  model.countParams())
model, startEpoch = models.attemptLoadModel(
    model          = model,
    checkpointDir  = workingDir)

if CUDA:
    model = model.cuda()

writer = SummaryWriter(workingDir)


if config['training']['optimizer'] == 'adam':
    lr        = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'sgd':
    lr        = config['training']['learning_rate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

epochLoss             = []
startSinceLastReport  = time.time()
wordsSinceLastReport  = 0
lossesSinceLastReport = []
bestMetric            = 0.0
bestEpoch             = 0
curMetric             = 0.0 # log perplexity or BLEU
numExamples           = min(len(src['content']), len(tgt['content']))
numBatches            = numExamples / batchSize



STEP = 0
for epoch in range(startEpoch, config['training']['epochs']):
    if curMetric > bestMetric:
        # rm old checkpoint
        for ckpt_path in glob.glob(workingDir + '/model.*'):
            os.system("rm %s" % ckpt_path)
        # replace with new checkpoint
        torch.save(model.state_dict(), workingDir + '/model.%s.ckpt' % epoch)

        bestMetric = curMetric
        bestEpoch  = epoch - 1

    losses = []
    for i in range(0, numExamples, batchSize):

        if args.overfit:
            i = 50

        batchIdx = i / batchSize
        
        inputContent, inputAux, outPut = data.minibatch(
            src, tgt, i, batchSize, maxLength, config['model']['model_type']
            )
        
        inputLinesSrc, _, srcLens, srcMask, _ = inputContent
        inputIdsAux, _, auxLens, auxMask, _ = inputAux
        inputLinesTgt, outputLinesTgt, _, _, _ = outPut
        decoderLogit, decoderProbs = model(inputLinesSrc, inputLinesTgt, srcMask, srcLens,
            inputIdsAux, auxLens, auxMask)

        optimizer.zero_grad()

        loss = lossCriterion(
            decoderLogit.contiguous().view(-1, tgtVocabSize), outputLinesTgt.view(-1)
        )

        losses.append(loss.item())
        lossesSinceLastReport.append(loss.item())
        epochLoss.append(loss.item())
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])

        writer.add_scalar('stats/grad_norm', norm, STEP)

        optimizer.step()

        if args.overfit or batchIdx % config['training']['batches_per_report'] == 0:

            s = float(time.time() - startSinceLastReport)
            eps = (batchSize * config['training']['batches_per_report']) / s
            avgLoss = np.mean(lossesSinceLastReport)
            info = (epoch, batchIdx, numBatches, eps, avgLoss, curMetric)
            writer.add_scalar('stats/EPS', eps, STEP)
            writer.add_scalar('stats/loss', avgLoss, STEP)
            logging.info('EPOCH: %s ITER: %s/%s EPS: %.2f LOSS: %.4f METRIC: %.4f' % info)
            startSinceLastReport = time.time()
            wordsSinceLastReport = 0
            lossesSinceLastReport = []


        STEP += 1
    if args.overfit:
        continue

    logging.info('EPOCH %s COMPLETE. EVALUATING...' % epoch)
    start = time.time()
    model.eval()
    
    devLoss = evaluation.evaluateLpp(model, srcTest, tgtTest, config)

    writer.add_scalar('eval/loss', devLoss, epoch)

    if args.bleu and epoch >= config['training'].get('inference_start_epoch', 1):
        curMetric, editDistance, inputs, preds, golds, auxs = evaluation.inferenceMetrics(
            model, srcTest, tgtTest, config)

        with open(workingDir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(workingDir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(workingDir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(workingDir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        writer.add_scalar('eval/edit_distance', editDistance, epoch)
        writer.add_scalar('eval/bleu', curMetric, epoch)

    else:
        cur_metric = devLoss

    model.train()

    logging.info('METRIC: %s. TIME: %.2fs CHECKPOINTING...' % (
        curMetric, (time.time() - start)))
    avgLoss = np.mean(epochLoss)
    epochLoss = []

writer.close()
