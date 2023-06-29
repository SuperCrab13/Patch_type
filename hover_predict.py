import torch
from importlib import import_module
from termcolor import colored
import numpy as np
from pathlib import Path
import glob
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import shutil
import os
from tqdm import tqdm
import argparse

type_info = {
    '0': 'no_label',
    '1': 'neoplastic',
    '2': 'inflammatory',
    '3': 'connective',
    '4': 'necros',
    '5': 'non_neoplastic_epithelial'
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/patches/SKCM/patches_lv0_ps256/DATA_DIRECTORY', help='directory of data')
parser.add_argument('--cancer_type', default='skcm')

args = parser.parse_args()

hovernet_config = {
  'nr_types': 6,
  'mode': 'fast',
  'hovernet_model_path': './data/weights/hovernet_fast_pannuke_type_tf2pytorch.tar',
  'type_info_path': 'type_info.json',
  'batch_size': 32}


class PatchData(Dataset):
    def __init__(self, wsi_path):
        """
        Args:
            data_24: path to input data
        """
        self.patch_paths = [p for p in wsi_path.glob("*")]
        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.GaussianBlur(kernel_size=3),
            # torchvision.transforms.RandomResizedCrop(size=256),
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = Image.open(self.patch_paths[idx]).convert('RGB')
        img = self.transforms(img)
        return img, os.path.split(self.patch_paths[idx])[1]


class Hovernet_infer:
    """ Run HoverNet inference """

    def __init__(self, config, dataloader):
        self.dataloader = dataloader

        method_args = {
            'method': {
                'model_args': {'nr_types': config['nr_types'], 'mode': config['mode'], },
                'model_path': config['hovernet_model_path'],
            },
            'type_info_path': config['type_info_path'],
        }
        run_args = {
            'batch_size': config['batch_size'],
        }

        model_desc = import_module("models.hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model")
        net = model_creator(**method_args['method']["model_args"])
        saved_state_dict = torch.load(method_args['method']["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
        net.load_state_dict(saved_state_dict, strict=False)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_infer = lambda input_batch: run_step(input_batch, net)

    def predict(self):
        output_list = []
        features_list = []
        path_list = []
        for idx, (data, path) in enumerate(tqdm(self.dataloader)):
            data = data.permute(0, 3, 2, 1)
            output, features = self.run_infer(data)
            # curr_batch_size = output.shape[0]
            # output = np.split(output, curr_batch_size, axis=0)[0].flatten()
            features_list.append(features)
            for out in output:
                if out.any() == 0:
                    output_list.append(0)
                else:
                    out = out[out != 0]
                    max_occur_node_type = np.bincount(out).argmax()
                    output_list.append(max_occur_node_type)
            path_list += path
        return output_list, np.concatenate(features_list), path_list


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                    "%s: Detect checkpoint saved in data-parallel mode."
                    " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


if __name__ == '__main__':
    data_path = args.data_dir
    patch_paths = glob.glob(data_path + r'*/*')
    hover_net_config = hovernet_config
    for i, wsi_input in enumerate(patch_paths):
        wsi_name = wsi_input.split('\\')[-1]
        print(f'Processing {i + 1} / {len(patch_paths)} patch')
        data = Path(wsi_input)
        if len(os.listdir(data)) == 0:
            print('empty dir for {}'.format(wsi_name))
            continue
        dataset = PatchData(data)
        dataloader = DataLoader(
            dataset,
            num_workers=0,
            batch_size=8,
            shuffle=False
        )
        hover_net = Hovernet_infer(hover_net_config, dataloader)
        node_type, feature, patch_list = hover_net.predict()
        # print(node_type)

        output = dict(zip(patch_list, node_type))
        print('saving {}... \n'.format(wsi_input))
        # with open(r'D:\code\WSI-HGNN-main\patch_label\{}.pkl'.format(wsi_input), 'wb') as f:
        #     pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

        if not os.path.exists('./patch_label/{}'.format(args.cancer_type)):
            os.mkdir('./patch_label/{}'.format(args.cancer_type))

        if not os.path.exists('./patch_label/{}/{}'.format(args.cancer_type, wsi_name)):
            os.mkdir('./patch_label/{}/{}'.format(args.cancer_type, wsi_name))

        if len(os.listdir('./patch_label/{}/{}'.format(args.cancer_type, wsi_name))) == 0:
            for k in type_info.keys():
                os.mkdir('./patch_label/{}/{}/{}'.format(args.cancer_type, wsi_name, type_info[k]))

        for patch in patch_list:
            shutil.copyfile(data_path+'/{}/{}'.format(wsi_name, patch),
                        './patch_label/{}/{}/{}/{}'.format(args.cancer_type, wsi_name, type_info[str(output[patch])], patch))
        print('{} finished'.format(wsi_name))
