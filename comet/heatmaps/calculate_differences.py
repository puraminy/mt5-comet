from transformers import T5ForConditionalGeneration
from transformers import T5Config
from torch import linalg as LA
import torch
from collections import defaultdict
import math
import statistics
import gc, os
from pathlib import Path
import click
import shutil

num_layers = 12
def sim_matrix(a, b, eps=1e-10):
    total = 0
    for dim in range(a.shape[0]):
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        total += math.acos(min(float(cos(a[dim], b[dim]).item()), 1)) / 3.141
    return total / a.shape[0]

def process_results(store):
    keys = ['q', 'k', 'v', 'o', 'wi', 'wo', 'xq', 'xk', 'xv', 'xo']
    for key in keys:
        if key in store:
            for idx in range(num_layers):
                store[key][idx] = store[key][idx]
            store[key] = store[key][:num_layers]
    
def get_norm(mat, n=None):
    if n == None:
        return torch.tensor(LA.norm(mat).item())
    return torch.tensor(LA.norm(mat, n).item())

def calc_l1(path, trials, fid):
    results_encoder = defaultdict(list)
    results_decoder = defaultdict(list)
    table_file_decoder = open(f'{path}/l1_decoder.htsv', 'w')
    table_file_encoder = open(f'{path}/l1_encoder.htsv', 'w')
    for dirname in trials:
        gc.collect()
        config = T5Config.from_pretrained(f'{dirname}')
        model = T5ForConditionalGeneration.from_pretrained(f'{dirname}', config=config)

        org_config = T5Config.from_pretrained(f'/home/pouramini/pret/{fid}/')
        org_model = T5ForConditionalGeneration.from_pretrained(f'/home/pouramini/pret/{fid}/', config=org_config)

        org_dict = org_model.state_dict()
        trained_dict = model.state_dict()

        for encoder_n in range(num_layers):
            q_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            results_encoder['q'].append(get_norm(q_org - q_new, 1))

            k_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            results_encoder['k'].append(get_norm(k_org - k_new, 1))

            v_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            results_encoder['v'].append(get_norm(v_org - v_new, 1))

            o_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            results_encoder['o'].append(get_norm(o_org - o_new, 1))

            if fid == "t5-v1" or fid == "t5-lmb":
                wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi_0.weight']
                wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi_0.weight']
            else:
                wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
                wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
            results_encoder['wi'].append(get_norm(wo_org - wo_new, 1)) # sic

            wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            results_encoder['wo'].append(get_norm(wo_org - wo_new, 1))

        for decoder_n in range(num_layers):
            q_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            results_decoder['q'].append(LA.norm(q_org - q_new, 1))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            results_decoder['k'].append(LA.norm(k_org - k_new, 1))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            results_decoder['v'].append(LA.norm(v_org - v_new, 1))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            results_decoder['o'].append(LA.norm(o_org - o_new, 1))

            q_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            results_decoder['xq'].append(LA.norm(q_org - q_new, 1))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            results_decoder['xk'].append(LA.norm(k_org - k_new, 1))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            results_decoder['xv'].append(LA.norm(v_org - v_new, 1))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            results_decoder['xo'].append(LA.norm(o_org - o_new, 1))

            if fid == "t5-v1" or fid == "t5-lmb":
                wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi_0.weight']
                wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi_0.weight']
            else:
                wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
                wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
            results_decoder['wi'].append(LA.norm(wo_org - wo_new, 1)) # sic

            wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            results_decoder['wo'].append(LA.norm(wo_org - wo_new, 1))

    process_results(results_decoder)
    process_results(results_encoder)
    for item in results_decoder:
        st = "\t".join([str(float(f.item())) for f in results_decoder[item]])
        table_file_decoder.write(f'{item}\t{st}\n')

    for item in results_encoder:
        st = "\t".join([str(float(f.item())) for f in results_encoder[item]])
        table_file_encoder.write(f'{item}\t{st}\n')

def calc_cosim(path, trials, fid):
    results_encoder = defaultdict(list)
    results_decoder = defaultdict(list)
    table_file_decoder = open(f'{path}/cossim_decoder.htsv', 'w')
    table_file_encoder = open(f'{path}/cossim_encoder.htsv', 'w')

    for dirname in trials:
        config = T5Config.from_pretrained(f'{dirname}')
        model = T5ForConditionalGeneration.from_pretrained(f'{dirname}', config=config)

        org_config = T5Config.from_pretrained(f'/home/pouramini/pret/{fid}/')
        org_model = T5ForConditionalGeneration.from_pretrained(f'/home/pouramini/pret/{fid}/', config=org_config)

        org_dict = org_model.state_dict()
        trained_dict = model.state_dict()

        for encoder_n in range(num_layers):
            q_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.q.weight']
            results_encoder['q'].append(sim_matrix(q_org, q_new))

            k_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.k.weight']
            results_encoder['k'].append(sim_matrix(k_org, k_new))

            v_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.v.weight']
            results_encoder['v'].append(sim_matrix(v_org, v_new))

            o_org = org_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'encoder.block.{encoder_n}.layer.0.SelfAttention.o.weight']
            results_encoder['o'].append(sim_matrix(o_org, o_new))

            if fid == "t5-v1" or fid == "t5-lmb":
                wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi_0.weight']
                wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi_0.weight']
            else:
                wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
                wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wi.weight']
            results_encoder['wi'].append(sim_matrix(wo_org, wo_new))

            wo_org = org_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'encoder.block.{encoder_n}.layer.1.DenseReluDense.wo.weight']
            results_encoder['wo'].append(sim_matrix(wo_org, wo_new))

        for decoder_n in range(num_layers):
            q_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.q.weight']
            results_decoder['q'].append(sim_matrix(q_org, q_new))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.k.weight']
            results_decoder['k'].append(sim_matrix(k_org, k_new))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.v.weight']
            results_decoder['v'].append(sim_matrix(v_org, v_new))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.0.SelfAttention.o.weight']
            results_decoder['o'].append(sim_matrix(o_org, o_new))

            q_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            q_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.q.weight']
            results_decoder['xq'].append(sim_matrix(q_org, q_new))

            k_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            k_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.k.weight']
            results_decoder['xk'].append(sim_matrix(k_org, k_new))

            v_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            v_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.v.weight']
            results_decoder['xv'].append(sim_matrix(v_org, v_new))

            o_org = org_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            o_new = trained_dict[f'decoder.block.{decoder_n}.layer.1.EncDecAttention.o.weight']
            results_decoder['xo'].append(sim_matrix(o_org, o_new))

            if fid == "t5-v1" or fid == "t5-lmb":
                wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi_0.weight']
                wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi_0.weight']
            else:
                wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
                wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wi.weight']
            results_decoder['wi'].append(sim_matrix(wo_org, wo_new))

            wo_org = org_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            wo_new = trained_dict[f'decoder.block.{decoder_n}.layer.2.DenseReluDense.wo.weight']
            results_decoder['wo'].append(sim_matrix(wo_org, wo_new))

    process_results(results_decoder)
    process_results(results_encoder)
    for item in results_decoder:
        st = "\t".join([str(float(f)) for f in results_decoder[item]])
        table_file_decoder.write(f'{item}\t{st}\n')

    for item in results_encoder:
        st = "\t".join([str(float(f)) for f in results_encoder[item]])
        table_file_encoder.write(f'{item}\t{st}\n')


@click.command()
@click.argument("fname", nargs=-1, type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--fid",
    "-fid",
    default="t5-base",
    type=str,
    help=""
)
@click.option(
    "--recalc",
    "-rc",
    is_flag=True,
    help=""
)
def main(fname, path, fid, recalc):
    global num_layers
    num_layers = 12
    fid = fid.replace("model_id_","").replace("_","-")
    print("fid:", fid)
    if fid == "t5-large":
        num_layers = 24
    if fid == "t5-base" or "t5-lmb":
        num_layers = 12
    elif fid == "t5-small":
        num_layers = 6

    if not fname:
        print("No file name provided")
        return
    else:
        if len(fname) > 1:
            print("len of input files is more than one")
            dirnames = list(fname)
        else:
            fname = fname[0]
            print("Searching ...")
            dirnames = []
            for root, dirs, _files in os.walk(path):
                for _file in _files:
                    _dir = str(Path(os.path.join(root, _file)).parent)
                    root_file = os.path.join(root,_file)
                    if all(s in root_file for s in fname.split("+")):
                        dirnames.append(_dir)
        #print("files: ",files)
    print("l1")
    common = os.path.commonprefix(dirnames) 
    for _dir in dirnames:
        table_file_decoder = f'{_dir}/l1_decoder.htsv'
        if Path(table_file_decoder).is_file() and not recalc:
            print("File already exits")
            continue
        print(_dir)
        _fid = fid
        if fid == "dir":
            if "t5-base" in _dir:
                _fid = "t5-base"
            if "t5-v1" in _dir:
                _fid = "t5-v1"
            if "t5-lmb" in _dir:
                _fid = "t5-lmb"
        try:
            calc_l1(_dir, [_dir], _fid)
        except:
            print("Error occoured")
    print("cosim")
    for _dir in dirnames:
        table_file_decoder = f'{_dir}/cossim_decoder.htsv'
        if Path(table_file_decoder).is_file() and not recalc:
            print("File already exits")
            continue
        print(_dir)
        _fid = fid
        if fid == "dir":
            if "t5-base" in _dir:
                _fid = "t5-base"
            if "t5-v1" in _dir:
                _fid = "t5-v1"
            if "t5-lmb" in _dir:
                _fid = "t5-lmb"
        try:
            calc_cosim(_dir, [_dir], _fid)
        except:
            print("Error occoured")


if __name__ == "__main__":
    main()
