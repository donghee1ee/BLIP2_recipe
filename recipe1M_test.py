import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import numpy as np
import os
from sklearn.metrics import f1_score
import json
import pickle
from keras.utils import to_categorical
from lavis.datasets.data_utils import Vocabulary
from tqdm import tqdm
from data_loader import get_loader

CUDA_LAUNCH_BLOCKING=1

def get_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # get the set of unique labels that appear in both lists
    labels = set(y_true) | set(y_pred)

    # calculate the true positives, false positives, and false negatives for each label
    tp = np.array([np.sum((y_true == label) & (y_pred == label)) for label in labels])
    fp = np.array([np.sum((y_true != label) & (y_pred == label)) for label in labels])
    fn = np.array([np.sum((y_true == label) & (y_pred != label)) for label in labels])

    # calculate the precision, recall, and F1 score for each label
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def get_iou(y_true, y_pred):
    gt_set = set(y_true)
    pred_set = set(y_pred)

    # find the intersection and union of the two sets
    intersection = gt_set.intersection(pred_set)
    union = gt_set.union(pred_set)

    # calculate the IoU or Jaccard score
    iou = len(intersection) / (len(union)+1e-06)

    return iou

def recipe1M_test():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

    vis_processors.keys()

    ## finetuned: /nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230516075/checkpoint_9.pth
    ## from scratch: /nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230515184/checkpoint_9.pth

    # - epoch 30 (from scratch): Pretrain_stage1/20230518080, Pretrain_stage2/20230519073/checkpoint_9.pth (epoch10)
    # - epoch 30 (fintuned) : Pretrain_stage1/20230518211, Pretrain_stage2/20230520080/checkpoint_19.pth (epoch20)

    model.load_checkpoint('/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230525102/checkpoint_4.pth')
    version = 'ingr_only'

    model.eval()
    print("Load Done. version: ", version)

    ingrs_vocab_v = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
    ingr2idx = ingrs_vocab_v.word2idx

    data_loaders = {}
    datasets = {}

    aux_data_dir = '/nfs_share2/code/donghee/inversecooking/data'
    data_dir = '/nfs_share2/code/donghee/inversecooking/data'
    maxseqlen = 15
    maxnuminstrs = 10
    maxnumlabels=20
    maxnumims = 5
    transform = vis_processors
    batch_size = 48 ##
    num_workers = 4
    max_num_samples = -1

    for split in ['test']:

        data_loaders[split], datasets[split] = get_loader(data_dir, aux_data_dir, split,
                                                            maxseqlen,
                                                            maxnuminstrs,
                                                            maxnumlabels,
                                                            maxnumims,
                                                            transform, batch_size,
                                                            shuffle=split == 'train', num_workers=num_workers,
                                                            drop_last=True,
                                                            max_num_samples=max_num_samples,
                                                            use_lmdb=True,
                                                            suff='')


    split = 'test'
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()
    # ingr_vocab_size = 1488
    
    f1s = []
    ious = []
    for loader in tqdm(data_loaders['test']):
        img_inputs, captions, ingr_gt, recipe_ids, img_id = loader
        prediction = model.generate({'image': img_inputs})
        
        for i in range(len(prediction)):
            pred = prediction[i].split('made with ')[-1].replace(', ', ',').replace(' ','_')
            pred = pred.split(',')
            if len(pred[-1]) == 0:
                pred = pred[:-1]
            pred = list(set(pred))

            pred_id = []
            for ingr in pred:
                try:
                    pred_id.append(ingr2idx[ingr])
                except:
                    print("no matching ingredient")
                    continue
            pred_id = list(set(pred_id))

            gt = ingr_gt[i]
            gt = gt.cpu().numpy()

            y_true = to_categorical(gt, ingr_vocab_size-1)
            y_pred = to_categorical(pred_id, ingr_vocab_size-1)
            y_true = np.sum(y_true, axis=0)
            y_pred = np.sum(y_pred, axis=0)

            f1 = f1_score(y_true, y_pred)
            f1s.append(f1)
            # im2ingr[img_id][name]['f1'] = f1

            iou = get_iou(gt, pred_id)
            ious.append(iou)
    
    mean_iou = sum(ious) / len(ious)
    mean_f1 = sum(f1s) / len(f1s)

    metrics = dict()
    metrics[version] = {
        'iou': mean_iou,
        'f1': mean_f1
    }
    print("mean_iou: ", mean_iou)
    print("mean_f1: ", mean_f1)

    with open("/nfs_share2/code/donghee/LAVIS/recipe1M_result2.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    print("==== Done ====")

def eval():
    original_images = []
    images = []
    paths = ['/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/2/6/9126297c7d.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/2/7/91273289fb.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/3/8/913851905f.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/5/1/915114360b.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/2/0002839c83.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/3/0003967721.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/0004a1d74e.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/0004d32dec.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/00049b8b85.jpg']
    ids = []
    for p in paths:
        image = Image.open(p).convert('RGB')
        images.append(image)
        original_images.append(image)
        ids.append(p.split('/')[-1])

    print(len(images))

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
    )


if __name__ == '__main__':
    recipe1M_test()


