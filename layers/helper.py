import torch

import nemo
from nemo.collections.asr.metrics import classification_accuracy, word_error_rate

logging = nemo.logging


def monitor_classification_triplet_training_progress(tensors: list, eval_metric=None, tb_logger=None):
    """
    Computes the top k classification accuracy of the model being trained.
    Prints sample to screen, computes and  and logs a list of top k accuracies
    to console and (optionally) Tensorboard
    Args:
      tensors: A list of 3 tensors (loss, logits, targets)
      eval_metric: An optional list of integers detailing Top@`k`
        in the range [1, max_classes]. Defaults to [1] if not set.
      tb_logger: Tensorboard logging object
    Returns:
      None
    """
    if eval_metric is None:
        eval_metric = [1]

    if type(eval_metric) not in (list, tuple):
        eval_metric = [eval_metric]

    top_k = eval_metric

    with torch.no_grad():
        logits, targets = tensors[1:3]
        topk_acc = classification_accuracy(logits, targets, top_k=top_k)

    tag = 'training_batch_top@{0}'
    logging.info(f'Loss: {tensors[0]}')
    logging.info(f'Triplet Loss: {tensors[-1]}')

    for k, acc in zip(top_k, topk_acc):
        if tb_logger is not None:
            tb_logger.add_scalar(tag.format(k), acc)

        logging.info(f"{tag.format(k)}: {acc * 100.: 3.4f}")


def monitor_triplet_encoder_training_progress(tensors: list, eval_metric=None, tb_logger=None):
    """
    Computes the top k classification accuracy of the model being trained.
    Prints sample to screen, computes and  and logs a list of top k accuracies
    to console and (optionally) Tensorboard
    Args:
      tensors: A list of 3 tensors (loss, logits, targets)
      eval_metric: An optional list of integers detailing Top@`k`
        in the range [1, max_classes]. Defaults to [1] if not set.
      tb_logger: Tensorboard logging object
    Returns:
      None
    """
    logging.info(f'Triplet Loss: {tensors[0]}')


def __gather_losses(losses_list: list) -> list:
    return [torch.mean(torch.stack(losses_list))]


def process_encoder_evaluation_batch(tensors: dict, global_vars: dict):
    """
    Creates a dictionary holding the results from a batch of samples
    """
    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'batchsize' not in global_vars.keys():
        global_vars['batchsize'] = []

    if 'embeds' not in global_vars.keys():
        global_vars['embeds'] = []

    if 'labels' not in global_vars.keys():
        global_vars['labels'] = []

    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('label'):
            labels = torch.cat(v, 0)  # if len(v) > 1 else v
        elif 'output' in kv:
            embeds = torch.cat(v, 0)
            if len(embeds.shape) == 2:
                embeds = embeds.unsqueeze(-1)

    batch_size = labels.size(0)
    global_vars['batchsize'] += [batch_size]
    global_vars['embeds'] += [embeds]
    global_vars['labels'] += [labels]


def process_encoder_evaluation_epoch(global_vars: dict, tag=None):
    """
    Calculates the aggregated loss and WER across the entire evaluation dataset
    """

    eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    batch_sizes = global_vars['batchsize']
    total_num_samples = torch.tensor(batch_sizes).sum().float()

    if tag is None:
        tag = ''

    logs = {f"Evaluation_Loss {tag}": eloss}

    logging.info(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")

    return logs
