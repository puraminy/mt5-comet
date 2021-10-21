# +
import sys
import csv
import glob
import click
import pandas as pd
import re
from os.path import expanduser
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from comet.utils.myfiles import *
from tabulate import tabulate
from pathlib import Path
import datetime
from comet.utils.myutils import *
import sacrebleu

from comet.evaluation.bleu.bleu import Bleu
from comet.evaluation.meteor.meteor_nltk import Meteor
from comet.evaluation.rouge.rouge import Rouge
from comet.evaluation.cider.cider import Cider
from collections import defaultdict
from argparse import ArgumentParser

import sys
import json
#reload(sys)
#sys.setdefaultencoding('utf-8')
from bert_score import score 

class BertScore:
    def __init__(self):
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        hyp_input = []
        ref_input = []
        same_indices = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            hyp_input += [hypo[0]] * len(ref)
            ref_input += ref
            same_indices.append(len(ref_input))

        p, r, f_scores = score(hyp_input, ref_input , model_type="bert-base-uncased")
 
        prev_idx = 0
        aggreg_f1_scores = []
        for idx in same_indices:
            aggreg_f1_scores.append(f_scores[prev_idx: idx].mean().cpu().item())
            prev_idx = idx

        return sum(aggreg_f1_scores)/len(aggreg_f1_scores), aggreg_f1_scores

    def method(self):
        return "Bert Score"

class QGEvalCap:
    def __init__(self, model_key, gts, res, results_file=None, calc_bert_score=False):
        self.gts = gts
        self.res = res
        self.results_file = results_file
        self.model_key = model_key
        self.cbs = calc_bert_score

    def evaluate(self):
        output = []
        scorers = [
#            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
#            (Cider(), "CIDEr"),
        ]
        if self.cbs:
            scorers.append((BertScore(), "Bert Score"))

        # =================================================
        # Compute scores
        # =================================================
        score_dict = {}
        scores_dict = {}
        #scores_dict["model_key"] = self.model_key
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    #print("%s: %0.5f"%(m, sc))
                    output.append(sc)
                    score_dict[m] = str(sc)
                    scores_dict[m] = list(scs)
            else:
                #print("%s: %0.5f"%(method, score))
                output.append(score)
                score_dict[method] = score
                scores_dict[method] = list(scores)
        if not self.cbs:
            score_dict["Bert Score"] = -1

        if self.results_file != None:
            with open(self.results_file, "a") as f:
                f.write(json.dumps(score_dict)+"\n")

        return score_dict, scores_dict

# -
def get_refs_preds(l, type, keys):
    keys = arg2dict(keys)
    head_key,tails_key,gens_key= keys["head"],keys["tails"],keys["gens"]
    if type==1:
        tails = l["fact"][tails_key]
        head = l["fact"][head_key]
        gens = l[gens_key]
        if "prompt" in l:
            prompt = l["prompt"]
            gens = [remove_prefix(g, prompt).strip() for g in gens]
    if type==2:
        tails = l[tails_key]
        head = l[head_key]
        gens = l[gens_key]
    if type==3:
        tails = [l[tails_key]]
        head = l[head_key]
        gens = l[gens_key]

    return gens, tails, head

def get2(l):
    if l:
        return list(zip(*l))[1]
    else:
        return 0


def topk_eval(out, data, data_type, k, keys, cbs=False):

    topk_gts = {}
    topk_res = {}
    topk_exact_match = 0 
    topk_exact_match_not_none = 0
    topk_bleu_score = []

    topk_is_head = 0
    print("len data", len(data))
    for i, l in enumerate(data):
        (gens, tails, head) = get_refs_preds(l, type=data_type, keys=keys)
        #print("Gens:", gens)
        #print("Tails:", tails)
        #print("Head:", head)
        new_tails = []
        for t in tails:
            end_index = t.index("[EOS]") if "[EOS]" in t else len(t)
            t = t[:end_index]
            new_tails.append(t)
        tails = new_tails

        sentence_tails = [t.lower().strip() for t in tails]
        #print("sentence_tails:", sentence_tails)
        split_tails = [t.lower().split() for t in tails]

        for (j, g) in enumerate(gens[:k]):
            #print("g:",g)
            #if g == "none" or g == "هیچ یک" or g == "<object>none</object>":
            #    continue
            start_index = g.index("[GEN]") if "[GEN]" in g else 0
            end_index = g.index("[EOS]") if "[EOS]" in g else 0
            if start_index > 0 and end_index > 0:
                g = g[start_index+5:end_index].strip()
            elif end_index > 0:
                g = g[:end_index].strip()
            #print("g2:",g)
            key = str(i) + "_" + str(j)
            topk_gts[key] = sentence_tails
            topk_res[key] = [g.lower()]
            
            #print("g.lower()", g.lower())
            #print("split_tails", split_tails)
            #print("g.lower().split", g.lower().split())

            #b = sacrebleu.sentence_bleu(sentence_tails, 
            #                  [g.lower()])
            #print("b1:",b.score)

            b = sentence_bleu(split_tails, 
                              g.lower().split(), 
                              weights=(0.5, 0.5))
            #print("b2:",b)
            
            topk_bleu_score.append((l, b))
            if g in sentence_tails:
                topk_exact_match += 1
                if g != "none" and g != "هیچ یک" and g != "<object>none</object>":
                    topk_exact_match_not_none += 1
            if g == head:
                topk_is_head += 1
    N = len(topk_gts)
    print("---------------TOP K={}---------------".format(k))
    print("Exact Match: {}  {:.2f}".format(topk_exact_match, topk_exact_match / N))
    print("Exact Match Not None {}  {:.2f}".format(topk_exact_match_not_none, topk_exact_match_not_none / N))
    #print("Mean sent BLEU score", np.mean(get2(topk_bleu_score)))
    print("len gts", len(topk_gts))
    QGEval = QGEvalCap(out, topk_gts, topk_res, calc_bert_score=cbs)
    scores,_ = QGEval.evaluate()
    scores["Exact_match"] = round(topk_exact_match / N,2)
    scores["Exact_match_not_none"] = round(topk_exact_match_not_none / N,2)
    scores["Mean sent BLEU score"] = np.mean(get2(topk_bleu_score))
    scores["Data rows"] = len(data)
    scores["Records"] = len(topk_gts)
    scores["TopK"] = k
    return scores


@click.command()
@click.argument("input_files_pattern", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--out",
    default="",
    type=str,
    help="The name for result csv file"
)
@click.option(
    "--data_type",
    default=2,
    type=int,
    help="The format of predictions file (refer to get_refs_preds function)"
)
@click.option(
    "--cbs",
    "-b",
    is_flag=True,
    help="Calculate Bert Score"
)
@click.option(
    "--ignore_inputs",
    "-i",
    is_flag=True,
    help="If there is no input file"
)
@click.option(
    "--task",
    default="",
    type=str,
    help=""
)
@click.option(
    "--clear",
    "-c",
    is_flag=True,
    help="It clears the last results and recalculate them"
)
@click.option(
    "--append",
    "-a",
    is_flag=True,
    help="It appends the new result to the end of output file"
)
@click.option(
    "--keys",
    default="head:head,tails:tails,gens:generations",
    type=str,
    help="A map for head, tails and generations keys, the default is: head:head,tails:tails,gens:generations"
)
@click.option(
    "--unicode",
    default="head",
    type=str,
    help="specify the field that needs unicode decoding (head, target)"
)
@click.option(
    "--from_to",
    default="",
    type=str,
    help="specify part of validation files eg. --100_1000"
)
def eval(path, input_files_pattern, out, data_type=2, topk=1, cbs=False, 
        ignore_inputs=False,
        task="",
        clear=False,
        append=True,
        keys="", unicode = "none", from_to=""):

    pred_inps = glob.glob(f"{path}/*{input_files_pattern}*predictions")
    if not pred_inps:
        print(f"No file was found using *{input_files_pattern}* pattern")
        return
    for pred_inp in pred_inps:
        pred_file = Path(pred_inp).name
        print(pred_file)
        fname = path + "/" + pred_file
        ext = Path(fname).suffix
        # split file path into folders
        P = list(reversed(fname.split("/")))
        p1 = P[1]
        p2 = P[2] if len(P) > 1 else ""
        p3 = P[3] if len(P) > 2 else ""
        res_exist = False
        out = out if out else p2 + "_" + p1
        if not out.endswith(".csv"):
           out += ".csv"
        if not "/" in out:
            out_fname = f"{path}/{out}"
        else:
            home = expanduser("~")
            out = out.replace("~/", home + "/")
            out_fname = out 

        if Path(out_fname).is_file() and not (clear or append):
            print(f"{out} already exists! use '--clear' or '--append' to reevaluate")
            return
        if ".json" in ext:
            ckp = re.findall(r"\d+", fname)[-1] #checkpoint_step
            pre = pred_file.split(ckp)[0]
            if task == "": task = pre
            print("Loading ", fname)
            if ext == ".jsonl":
                data = read_jsonl(path + "/" + pred_file)
            elif ext == ".json":
                data = read_json(path + "/" + pred_file)
            print("Len data:", len(data))
            scores = topk_eval(out, data, data_type, k=topk, keys=keys, cbs=cbs)
        else: 
            #when there are seperated input, predictions and targets files
            pred_found = True
            ckp = re.findall(r"\d+", fname)[-1] #checkpoint_step 
            pre = pred_file.split(ckp)[0] # task name before step
            if task == "": task = pre
            inps = glob.glob(f"{path}/{pre}inputs")
            inp_file = "None"
            inp_dict={}
            _from = 0
            _to = -1
            if from_to:
                _from = int(from_to.split("-")[0])
                _to = int(from_to.split("-")[1])
            if inps and not ignore_inputs:
                inp_file = inps[0]  
                with open(inp_file) as f:
                    input_lines = f.readlines()
                ii = _from
                if _to < 0: _to = len(input_lines)
                for inp in input_lines:
                    if ii >= _from and ii < _to:
                        inp_dict[inp] = {"targets": [], "gens": []}
                    ii += 1

            inps = glob.glob(f"{path}/{pre}targets")
            if not inps:
                print(f"!! target file {pre}targets for {pred_file} is missing!!!")
                continue
            target_file = inps[0]  
            with open(target_file) as f:
                target_lines = f.readlines()
            preds_file = f"{path}/{pred_file}"
            with open(preds_file, encoding="utf-8") as f:
                pred_lines = f.readlines()


            print("===================================")
            print(Path(inp_file).stem)
            print(Path(target_file).stem)
            print(Path(preds_file).stem)
            print("=========== Started ========================")
            _match = 0
            _match_none = 0
            if from_to:
                print("to:", _to)
                if inp_dict:
                    input_lines=input_lines[_from:_to]
                target_lines=target_lines[_from:_to]
                pred_lines=pred_lines[_from:_to]
            if not inp_dict:
                input_lines = [str(k) for k in range(len(target_lines))]
                inp_dict = {
                    str(k): {"targets": [], "gens": []} for k in range(len(target_lines))
                }
            for inp, target, pred in zip(input_lines, target_lines, pred_lines):
                inp_dict[inp]["targets"].append(target.strip())
                if not pred.strip() in inp_dict[inp]["gens"]:
                    inp_dict[inp]["gens"].append(pred.strip())
                if pred.strip() == "none" or pred.strip() == "هیچ یک":
                    _match_none += 1
                elif target.strip() == pred.strip():
                    inp2 = to_unicode(inp)
                    print(inp2)
                    print(target)
                    _match += 1

            print("input/target lines:", len(target_lines))
            print("len unique inputs:", len(inp_dict))
            print("ratio of unique inputs:{:.2f}".format(len(inp_dict)/len(target_lines)))
            print("Nones:{} {:.2f}".format(_match_none, _match_none/len(input_lines)))
            print("Matches:{} {:.2f}".format(_match, _match/len(input_lines)))
            cc = 1
            extra = ""
            res_name = pred_file + extra
            res_fname = f"{path}/{res_name}.jsonl"
            em = 0
            show_top = False
            kd = arg2dict(keys)
            head_key,tails_key,gens_key= kd["head"],kd["tails"],kd["gens"]
            with open(res_fname, "w", encoding="utf-8") as f:
                for head, val in inp_dict.items():
                    cc += 1
                    if unicode == "head":
                        head = to_unicode(head)
                    elif unicode == "tail":
                        val["targets"] = to_unicode(val["targets"])
                    if show_top and cc < 10 and gens[0] != "none":
                        print(head, ":", tails, "--", gens[0])
                    d = {}
                    d[head_key] = head.strip()
                    d[tails_key] = val["targets"]
                    d[gens_key] = val["gens"]
                    #print(cc, ": writing ", d)
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")

            data = read_jsonl(res_fname)
            #print("data:", data)
            scores = topk_eval(out, data, data_type = 2, k=1, keys=keys, cbs=cbs)
            # -
        # Saving Results
        print(scores)
        mydate = datetime.datetime.today()
        today = mydate.strftime("%Y-%m-%d")

        M = {}
        M["Task"] = task
        M["Checkpoint"] = ckp
        if not scores:
            print("No score")
        for key, val in scores.items():
            try:
                val = float(val)
                val = round(val, 3)
            except ValueError:
                pass
            M[key] = val

        for i, st in enumerate(task.split("_")):
            if st:
                M["T" + str(i)] = st
        M["Date"] = today
        M["P0"] = P[0] 
        for i, p in enumerate(P[:-3]):
            M["P" + str(i)] = p
        print("==============  CSV output =================")
        print(out_fname)
        print("============================================")
        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)
        if not Path(out_fname).is_file() or clear:
            with open(out_fname, "w") as f:  # You will need 'wb' mode in Python 2.x
                w = csv.DictWriter(f, M.keys())
                w.writeheader()
                w.writerow(M)
        elif append:
            with open(out_fname, "a") as f:  # You will need 'wb' mode in Python 2.x
                w = csv.DictWriter(f, fieldnames=M.keys())
                w.writerow(M)
            csv_df = pd.read_csv(out_fname,  error_bad_lines=False)
            final_df = csv_df.sort_values(by=['Exact_match_not_none'], ascending=False)
            #group = final_df.groupby("P0", as_index = False)
            #final_df = group.first() #.reset_index()
            final_df.to_csv(out_fname, index=False)

        #eval_file(path, pred_file, out, data_type, topk, cbs)




if __name__ == "__main__":
    eval()
 
