
from comet.train.common import *
from comet.train.template import *
from comet.train.data import *
import pandas as pd
from datetime import datetime
from pathlib import Path

#class MyDataLoader(datasets.Dataset):
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, split_df, split_name, method, 
            prompt_pos = "end", 
            rel_filter="", 
            num_samples=0, 
            ignore_blanks=False,
            only_blanks=False,
            inp_include="",
            inp_exclude="",
            targ_include="",
            targ_exclude="",
            pred_tresh=0,
            nli_group="all", per_record=False, per_prefix=False, is_even=False, start=0, 
            sampling=0, ex_type="",  samples_per_head=0, 
            save_ds_path="", repeat=1, pid=0, break_sent="", 
            sort_key="event", replace_blanks = False, 
            tokenizer=None, ph_num=3, limit_lang = False, 
            use_dif_templates=False, group_them=[], temp_num=1, 
            someone=False, match="", batch_size=0): 
        super(MyDataset).__init__()
        fingerprint = save_ds_path + "_" + split_name + "_"  + method + \
                "_" + str(len(split_df)) + "_" + str(num_samples) 
        self.flat_data = []
        self.data_split = {}
        self.sort_key = sort_key # sort index
        self.ph_num = ph_num
        self.someone = someone

        self.only_blanks = only_blanks
        self.samples_per_head = samples_per_head
        self.start = start
        self.pid = pid
        self.tokenizer = tokenizer
        self.prompt_pos = prompt_pos
        self.inp_include = inp_include
        self.inp_exclude = inp_exclude
        self.targ_include = targ_include
        self.targ_exclude = targ_exclude
        self.ex_type = ex_type
        self.per_record = per_record
        self.per_prefix = per_prefix
        self.is_even = is_even
        dlog.info("building query responses for {}".format(split_name))
        mlog.info(f"fill data input dataset len:{len(split_df)}")
        dlog.info(f"fill data input dataset len:{len(split_df)}")
        self.natural = inp_include == "natural"
        self.split_name = split_name
        self.rec_counter = 1
        self.method_index = 0
        if split_name != "train":
            self.start = 0
        if self.natural and split_name != "train": 
            self.natural = False 
        if self.natural:
            dlog.info("natural is ON")
        self.num_samples = num_samples
        self.replace_blanks = replace_blanks

        self.orig_df = None
        if group_them:
            self.orig_df = split_df.copy()
            split_df = split_df.groupby(group_them, as_index=False).first()
            dlog.info("*** Filtered for grouping_them %s ", group_them)

        if rel_filter == "all":
            rel_filter = ""
        if "@" in rel_filter:
            rel_filter, temp_num = rel_filter.split("@")
        self.temp_num = temp_num
        rel_filter = rel_filter.strip()
        if rel_filter:
            cond = ""
            op = "|"
            col = "prefix"
            for val in rel_filter.split("-"):
                assert val in all_rels, f"{val} is not in relations"
                cond += f"{op} (split_df['{col}'] == '{val}') "
            cond = cond.strip(op)
            split_df = split_df[eval(cond)]
            dlog.info("len after relation filter: %s", len(split_df))
        self.cats_num = cats_num = len(split_df["prefix"].unique())
        if self.per_prefix:
            self.num_samples = cats_num * num_samples
        use_all_data = False
        if self.num_samples == 0: 
            self.num_samples = len(split_df)
            self.samples_per_head = 0
            self.per_prefix = 0
            use_all_data = True
        for col in targets:
            if col in split_df:
                split_df[col] = split_df[col].astype(str)
        if ignore_blanks: # and len(split_df) > num_rows:
            #split_df = split_df[split_df["input_text"].str.contains('___')==False]
            split_df = split_df[split_df["target_text"] != "none"]
            dlog.info("*** Filtered for ignoring blanks or nones ")
        elif only_blanks:
            split_df = split_df[split_df["input_text"].str.contains('___')==True]
            #split_df = split_df[split_df["target_text"] != "none"]
            dlog.info("*** Filtered for including only with blanks ")
        if pred_tresh > 0 and "bert_score" in split_df:
            split_df = split_df[split_df["bert_score"] > pred_tresh]
            dlog.info("*** Filtered based on pred1 score higher than "+ pred_tresh)
        if nli_group != "all" and "nli_group" in split_df:
            split_df = split_df[split_df["nli_group"] == nli_group]
            dlog.info("*** Filtered based on nli_group "+ nli_group)
        if match: # and len(split_df) > num_rows:
            #split_df = split_df[split_df["input_text"].str.contains('___')==False]
            split_df = split_df[split_df["input_text"].str.match(match)]
            mlog.info("*** Filtered for match %s ", match)


        dlog.info(f"len after filtering:{len(split_df)}")
        mlog.info(f"len after filtering:{len(split_df)}")
        assert len(split_df) > 0, "Data frame is empty " + self.split_name
        self.num_records = self.num_samples
        if False:
            if self.num_samples < len(split_df) and not is_even: 
                #TODO 
                split_df = split_df.groupby("prefix").sample(n=self.num_samples)
                self.num_records = len(split_df)
                dlog.info(f"NUM samples %s, %s", self.num_samples, len(split_df))
                dlog.info(f"len after sampling:{len(split_df)}")

        split_df["freqs"] = split_df.groupby(['prefix','input_text'])['input_text'].transform('count')
        split_df = split_df.sort_values(by=["freqs","input_text", "prefix"], ascending=False)
        assert len(split_df) > 0, "Data frame is empty " + self.split_name + " " + str(self.num_samples)
        dlog.info("Num Samples: %s", self.num_samples)
        mlog.info("Num Samples: %s", self.num_samples)
        mlog.info("Cats Num: %s", cats_num)
        self.num_per_cat = self.num_samples // cats_num if cats_num > 1 else self.num_samples
        mlog.info("Num per cat: %s", self.num_per_cat)
        dlog.info("Num per cat: %s", self.num_per_cat)
        _spp = split_df[["prefix","input_text","target_text"]].groupby("prefix").count()
        for row in _spp.iterrows():
            mlog.info("%s : %s ", row[0], row[1].input_text)

        self.rel_counter = {}
        self.rel_filter = rel_filter
        self.lang_counter = {}
        self.sel_rels = []
        self.break_sent = break_sent
        if "other_rel" in ex_type:
            self.samples_per_head = 0
            self.num_per_cat = 0
            self.sel_rels = all_rels
            if "@" in ex_type:
                _rels = ex_type.split("@")[1]
                self.sel_rels = _rels.split("-")
        if rel_filter and not rel_filter in self.sel_rels:
            self.sel_rels.append(rel_filter)
        self.methods = method.split("+")
        if repeat < len(self.methods) - 1: 
            repeat = len(self.methods)
        if len(self.methods) > 1 and split_name != "train":
            self.methods = [self.methods[0]]
            repeat=1

        self.sampling = sampling
        self.batch_size = batch_size
        self.limit_lang = limit_lang
        self.split_df = split_df
        self.old_input = ""
        self.si = 0
        self.example_counter = 0
        self.use_dif_templates = use_dif_templates
        self.repeat = repeat
        mlog.info("Repeating for %s ", self.repeat)
        self.ex_df = pd.DataFrame()
        self._sels = self.sel_rels.copy()
        dlog.info("sels: %s", self._sels)
        self.save_path = save_ds_path  + fingerprint + ".pickle"
        if Path(self.save_path).is_file() and self.num_samples > 100_000 and not self.split_name == "sample":
            mlog.info("Loading from saved data %s ", self.save_path)
            self.load()
        else:
            _start = self.start
            _end = -1 #self.start + self.num_samples
            if self.flat_data:
                _data = self.flat_data
            else:
                if self.is_even or use_all_data:
                    _data = self.fill_all_data(_start, _end)
                else:
                    _data = self.fill_data(_start, _end, True, 
                        "".join(self.methods), self.num_samples, self.split_name,
                        self.cats_num, self.num_per_cat, self.ex_type, 
                        self.samples_per_head, 
                        self.inp_include, self.inp_exclude, 
                        self.targ_include, self.targ_exclude)
            if self.sort_key == "rep":
                _data = sorted(_data, key = lambda x:x[self.sort_key], reverse=True)
            elif self.sort_key != "none":
                _data = sorted(_data, key = lambda x:x[self.sort_key], reverse=False)
            self.flat_data = _data
            self.num_records = len(self.flat_data)

    def get_data(self):
        return self.flat_data

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, index):
        _item = self.flat_data[index]
        _ret = self.preproc(_item)
        return _ret

    def save(self):
        data = (self.flat_data, self.data_split)
        if not Path(self.save_path).exists():
            with open(self.save_path, "wb") as f:
                pickle.dump(data,f)
        else:
            mlog.info("The file already exists, skipping save ...")

    def save_data(self, save_path, merge=True, sample = 0):
        mlog.info("Saving data set in dataframe ...")
        df1 = None
        save_dir = save_path
        if save_path.endswith(".tsv"):
            save_dir = Path(save_path).parent
        if merge and Path(save_path).is_file():
            df1 = pd.read_table(save_path)
        rows = []
        ii = 0
        for item in self.flat_data:
            if item["rep"] != 0:
                continue
            ii += 1
            if sample > 0 and ii > sample:
                break

            row = item["row"]
            data = {"prefix":row["prefix"], "input_text":row["input_text"],"target_text":row["target_text"]}
            if df1 is None:
                rows.append(data)
            elif not ((df1["prefix"] == row["prefix"]) &
                 (df1["input_text"] == row["input_text"]) &
                 (df1["target_text"] == row["target_text"])).any():
                rows.append(data)
        df = pd.DataFrame(rows)
        #df.to_csv(os.path.join(save_dir, self.rel_filter + "_" + self.split_name + ".tsv"), sep="\t", index=False)
        if merge and df1 is not None:
            df = pd.concat([df1, df])
        df.to_csv(save_path, sep="\t", index=False)


    def load(self):
        with open(self.save_path, "rb") as f:
           data = pickle.load(f)
        self.flat_data, self.data_split = data
         
    def my_iter(self):
        iter_start = self.start
        iter_end = self.start+ self.num_samples
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = -1 #self.num_samples
        else:  # in a worker process
             # split workload
             assert self.num_samples > 0, "number of samples must be specified"
             per_worker = int(math.ceil((self.num_samples - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.num_samples)
        if self.flat_data:
            #self.flat_data = sorted(self.flat_data, key = lambda x:x[5])
            _data = self.flat_data
        else:
            if self.is_even:
                _data = self.fill_all_data(iter_start, iter_end)
            else:
               _data = self.fill_data(iter_start, iter_end, True, 
                    "".join(self.methods), self.num_samples, self.split_name,
                    self.cats_num, self.num_per_cat, self.ex_type, self.samples_per_head, 
                    self.inp_include, self.inp_exclude, 
                    self.targ_include, self.targ_exclude)
        mlog.info("Iter start: %s", iter_start)
        mlog.info("Iter end: %s", iter_end)
        self.flat_data = _data
        mlog.info("Len of flat data: %s", len(self.flat_data))
        self.num_records = len(self.flat_data)
        self.example_counter = 0
        _iter = iter(_data)
        return map(self.preproc, _iter)

    def preproc(self, data):
        event = data["event"]
        resp = data["resp"]
        rel = data["rel"]
        targ_col = data["targ_col"] if "targ_col" in data else "target_text"
        inp = data["inp"] if "inp" in data else "input_text"
        d = data["row"] if "row" in data else None
        context_df = data["context_df"] if "context_df" in data else None
        index = data["index"]
        self.rec_counter += 1
        rep = data["rep"]
        rel_token = rel_maps[rel]
        if self.natural:
            resp = resp.replace("PersonX intends", "")
            resp = resp.replace("PersonX قصد دارد", "")
        resp = resp.strip()
        gen_token = gen_tokens[targ_col]
        input_lang = "en" #langs[inp]
        target_lang = "en" #langs[targ_col]
        mt_idx = rep % len(self.methods)
        mt = self.methods[mt_idx]
        if "-fa" in mt and input_lang == "fa":
            event = toPers(event)
        if "-fa" in mt and target_lang == "fa":
            resp = toPers(resp)

        rt = RelTemplate(rel, self.temp_num)
        if rel and rel in template_conf:
           rt = template_conf[rel](rel, self.temp_num)

        qtemp, anstemp, ex_qtemp, ex_anstemp, context, flags = rt.create_templates(
                mt, index, 
                gen_pos="end", prompt_pos=self.prompt_pos, rel=rel)
        assert type(qtemp) == list, "qtemp must be a list"
        if self.use_dif_templates:
            _rep = rep
            if _rep > len(qtemp):
                _rep = len(qtemp) - 1
            qtemp = qtemp[_rep] 
        else:
            if self.pid < len(qtemp):
                qtemp = qtemp[self.pid] 
            else:
                qtemp = qtemp[-1] 
        #mbp("b")
        plen = relation_prompt_lengths[rel][0]
        mask = random.randint(0, plen-1)
        _qtemp = rt.fill_consts(qtemp, ex_qtemp, context, rel, d, context_df, mask=mask,method = mt, someone=self.someone)
        _anstemp = rt.fill_consts(anstemp, ex_anstemp, context,rel, d, context_df, mask=mask,method = mt, someone=self.someone)
        _query = rt.fill_vars(_qtemp, rel, event, resp, gen_token, 
                input_lang, target_lang, self.ph_num, self.temp_num, self.someone) 
        query = (index, _query)
        response = rt.fill_vars(_anstemp, rel, event, resp, gen_token, 
                input_lang, target_lang, self.ph_num, self.temp_num, self.someone)
        mbp("")
        __resp = response.replace(placeholder_token,"")
        _query = _query.strip()
        #mbp(_query)
        _q_len = len(_query.split(" "))
        sent = _query.replace(placeholder_token, __resp.strip())
        sent_split = sent.split(" ")
        if rep > 0 and self.break_sent:
            br = 0
            if self.break_sent == "person":
                indexes = [i for i,x in enumerate(sent_split) if x == "PersonX" or x == "others" or x == "PersonY"]
                if indexes:
                    br = indexes[-1]
                    _query = " ".join(sent_split[:br]) + " " + placeholder_token + " " + " ".join(sent_split[br+1:])
                    response = placeholder_token + " " + sent_split[br]
                    clog.info(">>>>>>>>>>>>>>>>>>>>=== rep:%s br:%s index:%s ========", rep, br, index)

            if (br == 0 and self.break_sent == "person") or self.break_sent == "random":  
                br = random.randint(len(sent_split)//2, len(sent_split)-1)
                if br > 0 and br < len(sent_split):
                    _query = " ".join(sent_split[:br]) + " " + placeholder_token
                    response = placeholder_token + " " + " ".join(sent_split[br:])
                    clog.info("================== rep:%s br:%s index:%s ========", rep, br, index)
            elif self.break_sent == "random_word":  
                _word = random.choice(sent_split)
                _query = _query.replace(_word, placeholder_token, 1)
                response = placeholder_token + " " + _word 
            elif self.break_sent == "random_span":  
                _start = random.randint(len(sent_split)//2, len(sent_split)-1)
                _end = random.randint(_start + 1, len(sent_split)-1)
                _phrase = " ".join(sent_split[_start:_end])
                _query = _query.replace(_phrase, placeholder_token, 1)
                response = placeholder_token + " " + _phrase 
            _q = _query.replace(">",">\n") 
            clog.info(_q)
            clog.info(response)
            clog.info("========================================")

        lang = input_lang + "2" + target_lang
        if self.example_counter < 10:
            clog.info(f"%%%%%%%%%%%%%%%%%% index:{index} rep:{rep} %%%%%%%%%%%%%%%%%%%")
            clog.info(sent)
            clog.info("---------------------------------------")
            clog.info(inp + "====>" + targ_col)
            _q = _query.replace(">",">\n") 
            clog.info(input_lang + ":"+ _q)
            clog.info(target_lang + ":" + response)
            self.example_counter += 1
        if self.replace_blanks and "___" in event:
            if "<extra_id_0>" in _query:
                _query = _query.replace("<extra_id_0>", "<extra_id_1>")
                _query = _query.replace("___", "<extra_id_0>")
                response = response.replace("<extra_id_0>", "<extra_id_1>")
                #response = "<extra_id_0> ___ " + response
            else:
                _query = _query.replace("___", "<extra_id_0>")
        #if not rel in self.data_split:
        #    self.data_split[rel] = {}
        #if not lang in self.data_split[rel]:
        #    self.data_split[rel][lang] = []
        #if query not in self.data_split[rel][lang]:
        #    self.data_split[rel][lang].append({query:[response]})
        #else:
        #    self.data_split[rel][lang][query].append(response)
        return {"query":_query, "event":event, "resp":response, "target":resp, "rel":rel, "index":self.rec_counter, "rep":rep, "flag":flags}
        #return (_query, event, response, rel, index, rep)


    def fill_all_data(self, iter_start, iter_end, show_progress=True):
        flat_data = []
        if iter_end < 0:
            iter_end = iter_start + self.num_samples
        kk = 0 
        dlog.info("========================== SPLIT: %s", self.split_name)
        dlog.info("get data from %s to %s", iter_start, iter_end)
        dlog.info("total rows: %s", len(self.split_df))
        context_df = None
        if show_progress:
            pbar = tqdm(total = self.num_samples, position=0, leave=True) #,dynamic_ncols=True)
            pbar.set_description("Preparing iterator "+ self.split_name)

        for index, d in self.split_df.iterrows():
            if kk < iter_start:
                dlog.info("!!!!!!!!! before start %s", iter_start)
                kk += 1
                continue
            rel = d["prefix"]
            inp = "input_text"
            targ_col = "target_text"
            event = d[inp]
            resp = d[targ_col]
            lang = "en2en"
            if kk > iter_end:
                self.flat_data.extend(flat_data)
                return flat_data
            _ditem = {"event":event, "resp":resp, 
                    "inp":inp, 
                    "targ_col":targ_col, 
                    "row":d,
                    "context_df":context_df,
                    "index": kk,
                    "rep":0,
                    "rel":rel}
            for rr in range(self.repeat):
                n_item = _ditem.copy()
                n_item["rep"] = rr
                flat_data.append(n_item)
            kk += 1
            if show_progress:
                pbar.update()
        self.flat_data.extend(flat_data)
        return flat_data

    @lru_cache
    def fill_data(self,  iter_start, iter_end, show_progress, 
            method, num_samples, split_name,
            cats_num, num_per_cat, ex_type, samples_per_head, 
            inp_include, inp_exclude, targ_include, targ_exclude):
        flat_data = []
        #if iter_end < 0:
        #    iter_end = iter_start + self.num_samples
        kk = 0
        jj = 0
        hh = 0
        dlog.info("==========NNNNN========= SPLIT: %s", self.split_name)
        dlog.info("get data from %s to %s", iter_start, iter_end)
        dlog.info("total rows: %s", len(self.split_df))
        context_rows=[]
        context_df = None
        old_event = ""
        filled_cat = {}
        if show_progress:
            pbar = tqdm(total = self.num_samples, position=0, leave=True) #,dynamic_ncols=True)
            pbar.set_description("Preparing iterator "+ self.split_name)

        no_new_item_added = False
        all_rows = len(self.split_df)
        ii = 0
        for index, d in self.split_df.iterrows():
            ii += 1 
            if kk < iter_start:
                dlog.info("!!!!!!!!! before start %s", iter_start)
                kk += 1
                continue
            rel = d["prefix"]
            if not rel in self.rel_counter:
                self.rel_counter[rel] = 0
            dlog.info("%s / %s --ii: %s, kk: %s jj:%s / %s )rel counter %s ", hh, self.num_samples, ii, kk, jj, all_rows, self.rel_counter)
            if len(filled_cat) == cats_num:
                dlog.info("all cats filled limit reached!")
                self.flat_data.extend(flat_data)
                return flat_data

            no_new_item_added = True
            if num_per_cat > 0 and self.rel_counter[rel] >= num_per_cat:
                dlog.info("!!!!!!!!! number per cat limit reached %s for %s", rel, num_per_cat)
                filled_cat[rel] = True
                continue 
            if "other_rel" in ex_type:
                if len(context_rows) >= len(self.sel_rels):
                    context_df = pd.DataFrame(data=context_rows)
                    self.ex_df = self.ex_df.append(context_df)
                    if self.rel_filter:
                        for item in context_rows:
                            if item["prefix"] == self.rel_filter:
                                d = item
                                rel = d["prefix"]
                    context_rows = []
                    self._sels = self.sel_rels.copy()
                else:
                    if (rel in self._sels and d["target_text"] != "none"): 
                        context_rows.append(d)
                        self._sels.remove(rel)
                    dlog.info("!!!!!!!!! just for including in conext rows %s", len(context_rows))
                    continue
            elif ex_type == "same_rel":
                context_df = self.split_df[self.split_df["prefix"] == rel].sample(n=int(self.sampling))
                clog.info("SAME rel for example type %s | %s ", self.sampling, len(context_df))

            elif ex_type:
                raise ValueError("Ex_type is invalid:" + ex_type)
            eng_inp = d["input_text"]
            self.si += 1
            if eng_inp != self.old_input:
                context_rows = []
                self._sels = self.sel_rels.copy()
                self.old_input = eng_inp
                self.si = 0
            elif samples_per_head > 0 and self.si >= samples_per_head:
                dlog.info("!!!!!!!!! samples per head limit %s", samples_per_head)
                continue
            for inp in inputs:
                if not inp in d or len(d[inp]) <= 1:
                    dlog.info("!!!!!!!!! not in dataset %s", inp)
                    continue
                if inp_include and not any(x in inp for x in inp_include.split("|")):
                    dlog.info("!!!!!!!!! not included input col %s", inp_include)
                    continue
                if inp_exclude and any(x in inp for x in inp_exclude.split("|")):
                    dlog.info("!!!!!!!!! excluded input col %s", inp_exclude)
                    continue
                input_lang = langs[inp]
                for targ_col in targets:
                    if not targ_col in d or len(d[targ_col]) <= 1:
                        dlog.info("!!!!!!!!! not target lang %s", targ_col)
                        continue
                    if targ_include and not any(x in targ_col for x in targ_include.split("|")):
                        dlog.info("!!!!!!!!! not included target col %s", targ_include)
                        continue
                    if targ_exclude and any(x in targ_col for x in targ_exclude.split("|")):
                        dlog.info("!!!!!!!!!  target exclude %s", targ_exclude)
                        continue
                    event = d[inp]
                    if event != old_event:
                        self.rel_counter[rel] += 1
                        hh += 1
                        if show_progress:
                            pbar.update()
                        old_event = event
                    resp = d[targ_col]
                    target_lang = langs[targ_col]
                    lang = input_lang + "2" + target_lang
                    if self.limit_lang:
                        if not lang in self.lang_counter:
                            self.lang_counter[lang] = 1
                        else:
                            self.lang_counter[lang] += 1
                        if (iter_end > 0 and self.lang_counter[lang] > iter_end):
                            dlog.info("Lang limit reached! %s %s", lang, self.lang_counter[lang])
                            self.flat_data.extend(flat_data)
                            return flat_data
                    no_new_item_added = False
                    _ditem = {"event":event, "resp":resp, 
                            "inp":inp, 
                            "targ_col":targ_col, 
                            "row":d,
                            "context_df":context_df,
                            "index": kk,
                            "rep":0,
                            "rel":rel}
                    for rr in range(self.repeat):
                        n_item = _ditem.copy()
                        n_item["rep"] = rr
                        flat_data.append(n_item)
                        jj += 1
                    kk += 1
                    if (iter_end > 0 and kk > iter_end):
                        dlog.info("record limit reached!")
                        self.flat_data.extend(flat_data)
                        return flat_data
            
        self.flat_data.extend(flat_data)
        dlog.info("!!! end of function %s %s %s", self.split_name, all_rows, self.num_samples)

        return flat_data

#tttttttttt
