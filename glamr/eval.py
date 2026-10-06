import argparse
import torch
import pickle
from tqdm import tqdm
from lib.utils.torch_utils import tensor_to
from utils.config import Config
from models.traj_pred_vae import TrajPredVAE
from lib.utils.tools import worker_init_fn
from data.assets.amass_dataset import AMASSEvalDataset, EMDBDataset
from torch.utils.data import DataLoader, Dataset


AMASS_DIRECTORY = 'datasets/amass_processed/v1'
EMDB_DIRECTORY = 'datasets/emdb_processed/v1'

def load_model(path):
    cfg = Config('traj_pred_demo', tmp=False, training=False)
    predictor = TrajPredVAE.load_from_checkpoint(path, cfg=cfg, args={})
    device = torch.device('cuda')
    predictor.to(device)
    predictor.eval()
    return predictor


def load_data(path, split='test', seq_len=100):
    test_dataset = AMASSEvalDataset(path, split, seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    return test_dataloader


def load_emdb_data(path, blend=0.0, seq_len=100):
    test_dataset = EMDBDataset(path, seq_len, blend)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    return test_dataloader


def run_checkpoint(checkpoint, saved_file, dataset):
    model = load_model(checkpoint)

    if dataset == 'emdb2':
        dataloader = load_emdb_data(EMDB_DIRECTORY)
    else:
        dataloader = load_data(AMASS_DIRECTORY)

    device = torch.device('cuda')
    test_result = {}
    save_keys =  ['infer_out_orient_q_tp', 
                  'infer_out_trans_tp',
                  'infer_out_swing_q_tp',
                  'infer_out_twist_tp',
                  'swing_q_tp',
                  'twist_tp', 'pose',
                  'context', 'orient_q_tp', 'trans_tp',
                  'eff_seq_len', 'idx']
    for i, batch in enumerate(tqdm(dataloader)):
        batch = tensor_to(batch, device)
        output = model.inference(batch, sample_num=1, recon=True, multi_step=False)
        saved_output = {}        
        for k in save_keys:
            if k in output.keys():
                saved_output[k] = output[k].detach().cpu()
        for k in ['seq_name', 'seq_ind']:
            saved_output[k] = output[k]
        saved_output['z'] = output['p_z_dist_infer'].mu.detach().cpu()
        test_result[i] = saved_output
    with open(saved_file, 'wb') as f:
        pickle.dump(test_result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='run')
    parser.add_argument('--path', default='glamr_train_result.ckpt')
    parser.add_argument('--save', default=None)
    parser.add_argument('--dataset', default='amass')

    args = parser.parse_args()

    if args.action == 'run':
        run_checkpoint(args.path, args.save, args.dataset)
