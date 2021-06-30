import math
import numpy as np
import sys
from collections import Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import editdistance

import src.data as data
from src.cuda import CUDA





def bleuStats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        sNgrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        rNgrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((sNgrams & rNgrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats



def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    logBleuPrec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + logBleuPrec)


def getBleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleuStats(hyp, ref))
    return 100 * bleu(stats)


def getEditDistance(hypotheses, reference):
    eDistance = 0
    for hyp, ref in zip(hypotheses, reference):
        eDistance += editdistance.eval(hyp, ref)

    return eDistance * 1.0 / len(hypotheses)
    


def decode_minibatch(max_len, start_id, model, src_input, srclens, srcmask,
        aux_input, auxlens, auxmask):
    """ argmax decoding """
    # Initialize target with <s> for every sentence
    tgt_input = Variable(torch.LongTensor(
        [
            [start_id] for i in range(src_input.size(0))
        ]
    ))
    if CUDA:
        tgt_input = tgt_input.cuda()

    for i in range(max_len):
        # run input through the model
        decoder_logit, word_probs = model(src_input, tgt_input, srcmask, srclens,
            aux_input, auxmask, auxlens)
        decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
        # select the predicted "next" tokens, attach to target-side inputs
        next_preds = Variable(torch.from_numpy(decoder_argmax[:, -1]))
        if CUDA:
            next_preds = next_preds.cuda()
        tgt_input = torch.cat((tgt_input, next_preds.unsqueeze(1)), dim=1)

    return tgt_input

def decode_dataset(model, src, tgt, config):
    """Evaluate model."""
    inputs = []
    preds = []
    auxs = []
    ground_truths = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()

        # get batch
        input_content, input_aux, output = data.minibatch(
            src, tgt, j, 
            config['data']['batch_size'], 
            config['data']['max_len'], 
            config['model']['model_type'],
            is_test=True)
        input_lines_src, output_lines_src, srclens, srcmask, indices = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_lines_tgt, output_lines_tgt, _, _, _ = output

        # TODO -- beam search
        tgt_pred = decode_minibatch(
            config['data']['max_len'], tgt['tok2id']['<s>'], 
            model, input_lines_src, srclens, srcmask,
            input_ids_aux, auxlens, auxmask)

        # convert seqs to tokens
        def ids_to_toks(tok_seqs, id2tok):
            out = []
            # take off the gpu
            tok_seqs = tok_seqs.cpu().numpy()
            # convert to toks, cut off at </s>, delete any start tokens (preds were kickstarted w them)
            for line in tok_seqs:
                toks = [id2tok[x] for x in line]
                if '<s>' in toks: 
                    toks.remove('<s>')
                cut_idx = toks.index('</s>') if '</s>' in toks else len(toks)
                out.append( toks[:cut_idx] )
            # unsort
            out = data.unsort(out, indices)
            return out

        # convert inputs/preds/targets/aux to human-readable form
        inputs += ids_to_toks(output_lines_src, src['id2tok'])
        preds += ids_to_toks(tgt_pred, tgt['id2tok'])
        ground_truths += ids_to_toks(output_lines_tgt, tgt['id2tok'])
        
        if config['model']['model_type'] == 'delete':
            auxs += [[str(x)] for x in input_ids_aux.data.cpu().numpy()] # because of list comp in inference_metrics()
        elif config['model']['model_type'] == 'delete_retrieve':
            auxs += ids_to_toks(input_ids_aux, tgt['id2tok'])
        elif config['model']['model_type'] == 'seq2seq':
            auxs += ['None' for _ in range(len(tgt_pred))]

    return inputs, preds, ground_truths, auxs


def inferenceMetrics(model, src, tgt, config):
    """ decode and evaluate bleu """
    inputs, preds, goldStadart, auxs = decode_dataset(model, src, tgt, config)
    
    bleu         = getBleu(preds, goldStadart)
    editdistance = getEditDistance(preds, goldStadart)
    
    inputs = [' '.join(seq) for seq in inputs]
    preds = [' '.join(seq) for seq in preds]
    goldStadart = [' '.join(seq) for seq in goldStadart]
    auxs = [' '.join(seq) for seq in auxs]

    return bleu, editdistance, inputs, preds, goldStadart, auxs
    
    



def evaluateLpp(model, src, tgt, config):
    """ evaluate log perplexity WITHOUT decoding
        (i.e., with teacher forcing)
    """
    weightMask = torch.ones(len(tgt['tok2id']))
    weightMask[tgt['tok2id']['<pad>']] = 0
    lossCriterion = nn.CrossEntropyLoss(weight=weightMask)
    if CUDA:
        weightMask    = weightMask.cuda()
        lossCriterion = lossCriterion.cuda()
    losses = []
    
    for i in range(0, len(src["data"]), config['data']['batch_size']):
        sys.stdout.write("\r%s/%s..." % (j, len(src['data'])))
        sys.stdout.flush()
        
        
        # get batch
        inputContent, inputAux, output = data.miniBatch(src, tgt, i, config['data']['batch_size'], config['data']['max_len'], config['model']['model_type'], isTest=True)
        inputLinesSrc, _, srcLens, srcMask, _  = inputContent
        inputIdsAux, _, auxLens, auxMask, _    = inputAux
        inputLinesTgt, outputLinesTgt, _, _, _ = output

        decoderLogit, decoderProbs = model(
            inputLinesSrc, inputLinesTgt, srcMask, srcLens,
            inputIdsAux, auxLens, auxMask)

        loss = lossCriterion(
            decoderLogit.contiguous().view(-1, len(tgt['tok2id'])),
            outputLinesTgt.view(-1)
        )
        
        losses.append(loss.item())

    return np.mean(losses)