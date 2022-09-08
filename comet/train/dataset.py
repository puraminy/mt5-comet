
from comet.train.common import *
import pandas as pd
from datetime import datetime
from pathlib import Path

#class MyDataset(datasets.Dataset):
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
        if self.num_samples == 0: 
            self.num_samples = len(split_df)
            self.samples_per_head = 0
            self.per_prefix = 0
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
                if self.is_even:
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

    def save_data(self, save_path, merge=True):
        mlog.info("Saving data set in dataframe ...")
        df1 = None
        if merge and Path(save_path).is_file():
            df1 = pd.read_table(save_path)
        rows = []
        for item in self.flat_data:
            if item["rep"] != 0:
                continue

            row = item["row"]
            data = {"prefix":row["prefix"], "input_text":row["input_text"],"target_text":row["target_text"]}
            if df1 is None:
                rows.append(data)
            elif not ((df1["prefix"] == row["prefix"]) &
                 (df1["input_text"] == row["input_text"]) &
                 (df1["target_text"] == row["target_text"])).any():
                rows.append(data)
        df = pd.DataFrame(rows)
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
        _counter = self.rec_counter
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

        qtemp, anstemp, ex_qtemp, ex_anstemp, context, flags = self.create_templates(
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
        _qtemp = self.fill_consts(qtemp, ex_qtemp, context, rel, d, context_df, mask=mask,method = mt, someone=self.someone)
        _anstemp = self.fill_consts(anstemp, ex_anstemp, context,rel, d, context_df, mask=mask,method = mt, someone=self.someone)
        _query = self.fill_vars(_qtemp, rel, event, resp, gen_token, 
                input_lang, target_lang, self.ph_num, self.temp_num, self.someone) 
        query = (index, _query)
        response = self.fill_vars(_anstemp, rel, event, resp, gen_token, 
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
        return {"query":_query, "event":event, "resp":response, "rel":rel, "index":_counter, "rep":rep, "flag":flags}
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
    def get_templates(self, method, index, gen_pos="end", prompt_pos="end"):
           ex_qtemp = ""
           ex_anstemp = ""
           qtemp = "{event}"
           anstemp = "{resp}"
           if method == "bart":
               qtemp = "{event} {rel} [GEN]"
               anstemp = "{resp}"
           elif method == "blank":
               qtemp = "{event} {rel_natural} {resp}"
               anstemp = "blank"
           elif method == "pred-emb":
               qtemp = "{enc_token_rest}"
               anstemp = "{ph} {enc_token_mask}"
           elif method == "pred-emb-rev":
               qtemp = "{enc_token_mask} {ph}"
               anstemp = "{ph} {enc_token_cont}"
           elif method == "rel-enc":
               qtemp = "{event} {rel_i} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "rel-dec":
               qtemp = "{event} {ph} {resp}"
               anstemp = "{ph} {rel_i}"
           elif method == "rel-mask":
               qtemp = "{event} {ph} {resp}"
               anstemp = "{ph} {rel_natural}"
           elif method == "unsup-rel":
               qtemp = "{event} {ph} {resp}"
               anstemp = "{ph} {prefix}"
           elif method == "unsup-wrap-rel":
               qtemp = "{com_i} {event} {ph} {resp}"
               anstemp = "{prefix}"
           elif method == "rel-mask-wrap":
               qtemp = "{enc_com_token} {event} {ph} {resp}"
               anstemp = "{ph} {rel_natural}"
           elif method == "rel-unsup":
               qtemp = "{event} {rel_natural} {ph} "
               anstemp = "{ph} {resp}"
           elif method == "sup-pred-enfa":
               qtemp = "{input_text} {rel_i} {gen_fa}"
               anstemp = "{input_text_fa} {dec_token} {target_text_fa}"
           elif method == "sup-enfa":
               qtemp = "{input_text} {rel_i} {target_text} {gen_fa}"
               anstemp = "{input_text_fa} {dec_token} {target_text_fa}"
           elif method == "sup-enmix":
               qtemp = "{input_text} {rel_i} {target_text} {gen}"
               anstemp = "{event} {dec_token} {gen} {resp}"
           elif method == "unsup-nat-gen":
               qtemp = "{rel_i} {event} {rel_natural} {gen} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-nat-tokens":
               qtemp = "{event} {nat_tokens}" 
               anstemp = "{resp} {end}"
           elif method == "unsup-nat-tokens":
               qtemp = "{event} {nat_tokens} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-nat":
               #qtemp = "{rel_natural_pure}" 
               qtemp = "{rel_natural}"
               anstemp = "{resp} {end}"
           elif method == "sup-nat-tail":
               #qtemp = "{rel_natural_pure}" 
               qtemp = "{rel_natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-nat":
               qtemp = "{rel_natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-nat-head":
               qtemp = "{rel_natural}"
               anstemp = "{resp} {end}"
           elif method == "unsup-nat-n":
               qtemp = "{rel_nat_n}"
               anstemp = "{ph} {resp} {end}"
           elif method == "enc-unsup-nat":
               qtemp = "{rel_i} {event} {rel_natural} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-nat-fa":
               qtemp = "{event} {rel_natural_fa} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-nat":
               qtemp = "{rel_i} {rel_natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-nat-end":
               qtemp = "{rel_natural} {rel_i}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-nat-mid":
               qtemp = "{event} {rel_i} {rel_natural} {rel_i} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-lang":
               qtemp = "{rel_i} {event} {rel_lang_i} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-gen":
               qtemp = "{event} {gen}"
               anstemp = "{resp} {end}"
           elif method == "sup-wrap":
               qtemp = "{rel_i_start} {event} {rel_i_end} {gen}"
               anstemp = "{resp} {end}"
           elif method == "sup-wrap-nat":
               qtemp = "{rel_i} {rel_natural}"
               anstemp = "{resp} {end}"
           elif method == "sup-no-gen":
               qtemp = "{event}"
               anstemp = "{resp} {end}"
           elif method == "gen":
               qtemp = "{gen}"
               anstemp = "{resp}"
           elif method == "pred-enfa":
               qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {ph} {event} {rel_natural} {rel_i_end} {gen_end} <extra_id_1>"
               anstemp = "{ph} {target_text} <extra_id_1> {resp} <extra_id_2>"
           elif method == "context-en":
               qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {target_text} {rel_i_start} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "context-faen":
               qtemp = "{rel_i_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_en} {target_text} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-n-example":
               qtemp = "{examples} {event} {rel_i} {gen} {ph}"
               ex_qtemp = "{input_text} {rel_i} {target_text} {end}"
               anstemp = "{ph} {resp} {end}"
           elif method == "trans":
               qtemp = "{target_text} en2fa"
               anstemp = "{target_text_fa}"
           elif method == "unsup-n-example":
               qtemp = "{examples} {gen} {ph}"
               ex_qtemp = "{gen} {input_text} {end} \n"
               anstemp = "{ph} {event} {end}"
           elif method == "story-wrap":
               qtemp = "{rel_i} {event} {rel_natural} {ph}"
               context = "{xAttr} {xIntent} {xReact}"
               ex_qtemp = "{rel_enc_token} {rel_natural_en} {target_text} {end}"
               anstemp = "{ph} {resp} {end}"
           elif method == "event-resp-n-wrap":
               qtemp = "{event} {examples} {rel_i} {event} {rel_natural} {ph}"
               ex_qtemp = "{rel_natural_en} {target_text} {end} \n"
               anstemp = "{ph} {resp} {end}"
           elif method == "gpt-event-resp-n-wrap":
               qtemp = "{examples} {rel_i}"
               ex_qtemp = "{rel_i} {input_text} {rel_natural_en} {target_text} {sep} \n"
               anstemp = "{event} {rel_natural} {resp} {end}"
           elif method == "fa-gpt-event-resp-n-wrap":
               qtemp = "{examples} {rel_i}"
               ex_qtemp = "{rel_i} {input_text_fa} {rel_natural_fa} {target_text_fa} {sep} \n"
               anstemp = "{event} {rel_natural} {resp} {end}"
           elif method == "unsup-wrap-n-example-fa":
               qtemp = "{examples} {gen} {ph}"
               ex_qtemp = "{gen} {input_text_fa} {end} \n"
               anstemp = "{ph} {event} {end}"
           elif method == "gpt-event-n":
               qtemp = "{examples} {gen}"
               ex_qtemp = "{gen} {input_text} {end} \n"
               anstemp = "{event} {end}"
           elif method == "gpt-fa-event-n":
               qtemp = "{examples} {gen}"
               ex_qtemp = "{gen} {input_text_fa} {end} \n"
               anstemp = "{event} {end}"
           elif method == "unsup-wrap-nat-example":
               qtemp = "{examples} {event} {rel_i} {ph}"
               ex_qtemp = "{input_text} {rel_natural} {target_text}. {end} \n"
               anstemp = "{ph} {resp} {end}"
           elif method == "gpt-n-example":
               qtemp = "{examples} {event} {rel_natural}"
               ex_qtemp = "{input_text} {rel_natural} {target_text} {end}"
               anstemp = "{resp} {end}"
           elif method == "gpt-wrap-n-example":
               qtemp = "{examples} {event} {rel_i}"
               ex_qtemp = "{input_text} {rel_i} {target_text} {end}"
               anstemp = "{resp} {end}"
           elif method == "unsup-wrap-context-n-dec":
               qtemp = "{event} {rel_i} {gen} {ph}"
               ex_qtemp = "{input_text} {target_text} {end}"
               anstemp = "{examples} {ph} {resp} {end}"
           elif method == "unsup-wrap-context-enfa":
               qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_fa} {target_text_fa} {rel_i_start} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-context-fa":
               qtemp = "{rel_i_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_fa} {target_text_fa} {rel_i_start} {event} {rel_natural} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "sup":
               qtemp = "{rel_token} {event}"
               anstemp = "{resp} {end}"
           elif method == "sup-no-rel":
               qtemp = "{event}"
               anstemp = "{resp} {end}"
           elif method == "sup-end":
               qtemp = "{event} {rel_token}" 
               anstemp = "{resp} {end}"
           elif method == "sup-wrap-gen":
               qtemp = "{rel_i_start} {gen_start} {event} {rel_i_end} {gen_end}"
               anstemp = "{resp} {end}"
           elif method == "gpt-wrap-tokens":
               qtemp = "{examples} {tokens} {event} "
               ex_qtemp = "{tokens} {input_text} {target_text} {end}"
               anstemp = "{resp} {end}"
           elif method == "gpt-wrap":
               qtemp = "{event} {rel_i}"
               anstemp = "{resp} {end}"
           elif method == "gpt":
               qtemp = "{event} {rel_natural}"
               anstemp = "{resp} {end}"
           elif method == "unsup-wrap-fw":
               qtemp = "{event} {rel_fw} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup":
               qtemp = "{event} {rel_token} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-gen":
               qtemp = "{rel_token} {event} {gen} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-com":
               qtemp = "{com_i} {event} {rel_i} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap":
               qtemp = "{rel_i_start} {event} {rel_i_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-start":
               qtemp = "{rel_i} {event} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-tokens" or method  == "sup-tokens-wrap":
               qtemp = "{event} {tokens}"
               anstemp = "{resp} {end}"
           elif method == "unsup-tokens" or method  == "unsup-tokens-wrap":
               qtemp = "{event} {tokens} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-tokens-nat":
               qtemp = "{tokens} {rel_natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-rel-nat":
               qtemp = "{prefix} {rel_natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-rel-ans":
               qtemp = "{event} {tokens} {ph} "
               anstemp = "{ph} {rel_natural_pure} {resp}"
           elif method == "unsup-nat-rel-ans":
               qtemp = "{event}. {ph}"
               anstemp = "{ph} {rel_natural_pure} {resp}."
           elif method == "unsup-tokens-rand" or method  == "unsup-tokens-rand-wrap":
               qtemp = "{event} {tokens-rand} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-tokens-start":
               qtemp = "{tokens} {event}"
               anstemp = "{resp} {end}"
           elif method == "unsup-tokens-start" or method == "unsup-tokens-wrap-start":
               qtemp = "{tokens} {event} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-tokens-gen":
               qtemp = "{event} {tokens} {gen_lang} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-gen":
               qtemp = "{rel_i_start} {gen_start} {event} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-dec":
               qtemp = "{rel_i_start} {gen_start} {event} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {dec_token} {resp} {end}"
           elif method == "unsup-wrap-2h":
               qtemp = "{rel_i} {event} {rel_i} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-2":
               qtemp = "{rel_i} {event} {ph} {rel_i} "
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-3":
               qtemp = "{rel_i} {gen_start} {event} {rel_i} {gen_end} {ph}"
               anstemp = "{ph} {dec_token} {resp} {end}"
           else:
               raise ValueError("not supprted method: " + method)
           flags = {
                   "method":method,
                   "wrap":"wrap" in method,
                   "freeze":"wrap" in method,
                   "unfreeze":False,
                   }
           return qtemp, anstemp, ex_qtemp, ex_anstemp, flags
           
    def create_templates(self, method, index, gen_pos="end", prompt_pos="end", rel=""):
           ctx = ["{" + x + "}" for x in all_rels]
           context = " ".join(ctx)
           mbp("")
           qtemp, anstemp, ex_qtemp, ex_anstemp, flags = self.get_templates(method, 
                    self.rec_counter, gen_pos=gen_pos, prompt_pos=prompt_pos)
           self.rec_counter += 1
           if not type(qtemp) == list:
               qtemp = [qtemp]
           ret_q = []
           for q in qtemp:
               while "  " in q:
                   q = q.replace("  "," ")
               if method.startswith("sup"):
                   q = q.replace("{ph}","")
                   q = q.replace("{rel_natural}","{rel_natural_pure}")
               q = fix_pos(q, gen_pos, prompt_pos)
               ret_q.append(q)
           qtemp = ret_q
           return qtemp, anstemp, ex_qtemp, ex_anstemp, context, flags

    def fill_consts(self, template, ex_temp, context,rel, row, rows=None, mask=-1, method="", someone=False):
        #dlog.info("fill consts, input text: %s", template)
        text = fill_const_for_rel(template, row)
        #dlog.info("fill consts, input text: %s", text)
        plen = relation_prompt_lengths[rel]
        text = text.replace("{ph}", placeholder_token)
        #dlog.info("fill consts, input text: %s", context)
        counter = 0
        pi = 0
        text = fill_prompt(text, rel, "{rel_i}")
        text = fill_prompt_regex(text, rel, "{([a-zA-Z]+)_(\d+)}")
        text = fill_prompt(text, "com", "{com_i}")
        text = fill_prompt(text, rel, "{tokens}")
        text = fill_prompt(text, rel, "{tokens-rand}")
        if "{examples}" in text:
            examples = ""
            assert rows is not None and len(rows) > 0, "Since there are examples in template, rows must be provided"
            ii = 1
            for idx, _row in rows.iterrows():
                example = ex_temp
                numbered = False
                if "{num}" in ex_temp:
                    numbered = True
                    example = example.replace("{num}", "")
                #dlog.info("example: %s", _row)
                if "{rel_i}" in ex_temp:
                    assert enc_prompt != "", "Prompt was not set!"
                example = fill_const_for_rel(example, _row)
                example = fill_prompt(example, rel, "{rel_i}")
                example = fill_prompt(example, rel, "{tokens}")
                example = fill_prompt(example, rel, "{tokens-rand}")
                for key,value in _row.items():
                    val = str(value)
                    if "fa" in method and "_fa" in key:
                        val = toPers(val)
                    example = example.replace("{" + key + "}", val)
                if numbered: 
                    examples += " " + str(ii) + ") " + example
                else:
                    examples += example
                ii += 1

            text = text.replace("{examples}", examples + " " + str(ii) + ")")
        ## storyyyy
        ii = 1
        if rows is not None:
            for idx, _row in rows.iterrows():
                example = ex_temp
                relation = _row["prefix"]
                if relation == rel:
                    continue
                numbered = False
                if "{num}" in ex_temp:
                    numbered = True
                    example = example.replace("{num}", "")
                if "{rel_i}" in ex_temp:
                    assert enc_prompt != "", "Prompt was not set!"
                example = fill_const_for_rel(example, _row)
                example = fill_prompt(example, relation, "{rel_i}")
                example = fill_prompt(example, "com", "{com_i}")
                for key,value in _row.items():
                    val = str(value)
                    if "fa" in method and "_fa" in key:
                        val = toPers(val)
                    example = example.replace("{" + key + "}", val)
                if numbered: 
                    example = " " + str(ii) + ") " + example
                context = context.replace("{" + relation + "}", example)
                ii += 1
        for relation in all_rels:
            if relation == rel:
                context = context.replace("{" + relation + "}", text)
            else:
                context = context.replace("{" + relation + "}", "")
        ii = 1
        while "{ph}" in context:
            ph = f"<extra_id_{ii}>"
            ii += 1
            context = context.replace("{ph}", ph, 1)

        #dlog.debug("after: %s", context)
        return context 


    def fill_vars(self, template, rel, event, resp, gen_token= "gen_en",  inp_lang="en", resp_lang="en", ph_num=3, temp_num = 1, someone=False):
        if type(temp_num) == str and temp_num.isnumeric():
            temp_num = int(temp_num)
        assert temp_num in rel_nat_maps[rel], rel + " for " + str(temp_num)
        event1, event2= "",""
        if "{@@@}" in event:
            #mbp("b")
            event1, event2 = event.split("{@@@}")
            event = event.replace("{@@@}"," ")
        rel_natural = rel_nat_maps[rel][temp_num]        
        rel_natural_tokens = rel_nat_maps[rel]["nat-tokens"]        
        rel_natural_pure = rel_natural.replace("{ph}", "")
        rel_natural_pure = rel_natural_pure.replace(".", "")
        rel_n = ""
        for i in range(ph_num):
            rel_n += "<extra_id_" + str(i) + "> "
        rel_nat_n = rel_natural.replace("{ph}", rel_n)
        rel_natural = rel_natural.replace("{ph}", placeholder_token)

        rep1  = {
                "{rel_natural}":rel_natural,
                "{rel_natural_pure}":rel_natural_pure,
                "{rel_nat_n}":rel_nat_n,
                "{nat_toekns}":rel_natural_tokens,
                "{gen}":gen_token}
        rep2  = {
                "{event}":event, 
                "{event1}":event1, 
                "{event2}":event2, 
                "{resp}":resp
                }
        rep1 = dict((re.escape(k), v) for k, v in rep1.items()) 
        rep2 = dict((re.escape(k), v) for k, v in rep2.items()) 
        pattern1 = re.compile("|".join(rep1.keys()))
        pattern2 = re.compile("|".join(rep2.keys()))
        text = pattern1.sub(lambda m: rep1[re.escape(m.group(0))], template)
        text = pattern2.sub(lambda m: rep2[re.escape(m.group(0))], text)
        if someone:
            text = text.replace("PersonX", "someone")
            text = text.replace("PersonY", "someone else")
            text = text.replace("PersonZ", "others")
        lang = resp_lang
        plen = relation_prompt_lengths[rel]
        text = fill_prompt(text, rel, "{rel_lang_i}", lang=lang)
        text = fill_prompt(text, rel, "{gen_lang}", lang=lang)
        return text

