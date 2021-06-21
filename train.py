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
src, tgt = data.readDataSet(
    srcFile=config['data']['src'],
    config=config,
    tgtFile=config['data']['tgt'],
    attributeVocab=config['data']['attribute_vocab'],
    useNgrams=config['data']['ngram_attributes']
)

srcTest, tgtTest = data.readDataSet(
    srcFile=config['data']['src_test'],
    config=config,
    tgtFile=config['data']['tgt_test'],
    attributeVocab=config['data']['attribute_vocab'],
    useNgrams=config['data']['ngram_attributes'],
    trainSrc=src,
    trainTgt=tgt
)

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
model, start_epoch = models.attemptLoadModel(
    model=model,
    checkpointDir=workingDir)

if CUDA:
    model = model.cuda()

writer = SummaryWriter(workingDir)


if config['training']['optimizer'] == 'adam':
    lr = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['learning_rate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")

epochLoss = []
startSinceLastReport = time.time()
wordsSinceLastReport = 0
lossesSinceLastReport = []
bestMetric = 0.0
bestEpoch = 0
curMetric = 0.0 # log perplexity or BLEU
numExamples = min(len(src['content']), len(tgt['content']))
numBatches = numExamples / batchSize

input()
aqui

STEP = 0
for epoch in range(start_epoch, config['training']['epochs']):
    if cur_metric > best_metric:
        # rm old checkpoint
        for ckpt_path in glob.glob(working_dir + '/model.*'):
            os.system("rm %s" % ckpt_path)
        # replace with new checkpoint
        torch.save(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)

        best_metric = cur_metric
        best_epoch = epoch - 1

    losses = []
    for i in range(0, num_examples, batch_size):

        if args.overfit:
            i = 50

        batch_idx = i / batch_size

        input_content, input_aux, output = data.minibatch(
            src, tgt, i, batch_size, max_length, config['model']['model_type'])
        input_lines_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output
        
        decoder_logit, decoder_probs = model(
            input_lines_src, input_lines_tgt, srcmask, srclens,
            input_ids_aux, auxlens, auxmask)

        optimizer.zero_grad()

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, tgt_vocab_size),
            output_lines_tgt.view(-1)
        )

        losses.append(loss.item())
        losses_since_last_report.append(loss.item())
        epoch_loss.append(loss.item())
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])

        writer.add_scalar('stats/grad_norm', norm, STEP)

        optimizer.step()

        if args.overfit or batch_idx % config['training']['batches_per_report'] == 0:

            s = float(time.time() - start_since_last_report)
            eps = (batch_size * config['training']['batches_per_report']) / s
            avg_loss = np.mean(losses_since_last_report)
            info = (epoch, batch_idx, num_batches, eps, avg_loss, cur_metric)
            writer.add_scalar('stats/EPS', eps, STEP)
            writer.add_scalar('stats/loss', avg_loss, STEP)
            logging.info('EPOCH: %s ITER: %s/%s EPS: %.2f LOSS: %.4f METRIC: %.4f' % info)
            start_since_last_report = time.time()
            words_since_last_report = 0
            losses_since_last_report = []

        # NO SAMPLING!! because weird train-vs-test data stuff would be a pain
        STEP += 1
    if args.overfit:
        continue

    logging.info('EPOCH %s COMPLETE. EVALUATING...' % epoch)
    start = time.time()
    model.eval()
    dev_loss = evaluation.evaluate_lpp(
            model, src_test, tgt_test, config)

    writer.add_scalar('eval/loss', dev_loss, epoch)

    if args.bleu and epoch >= config['training'].get('inference_start_epoch', 1):
        cur_metric, edit_distance, inputs, preds, golds, auxs = evaluation.inference_metrics(
            model, src_test, tgt_test, config)

        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        writer.add_scalar('eval/edit_distance', edit_distance, epoch)
        writer.add_scalar('eval/bleu', cur_metric, epoch)

    else:
        cur_metric = dev_loss

    model.train()

    logging.info('METRIC: %s. TIME: %.2fs CHECKPOINTING...' % (
        cur_metric, (time.time() - start)))
    avg_loss = np.mean(epoch_loss)
    epoch_loss = []

writer.close()
