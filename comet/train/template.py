from comet.train.common import *
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
import comet.train.mylogs as logs

class RelTemplate:
    def __init__(self, rel, temp_num=1):
        self.rel = rel
        self.temp_num = temp_num
        self.num_prompt_tokens = int(logs.main_args["prompt_token_num"]) #len(encoders) 
        self.num_prompts = int(logs.main_args["n_prompts"]) #len(encoders) 
        self.rec_counter = 1
        self.encoder_prompts = {} 
        self.decoder_prompts = {}
        self.common_tokens = []

    def get_templates(self, method, index, gen_pos="end", prompt_pos="end"):
           ex_qtemp = ""
           ex_anstemp = ""
           qtemp = "{event}"
           anstemp = "{resp}"
           tn = int(self.temp_num)
           if method == "bart":
               qtemp = "{event} {rel} [GEN]"
               anstemp = "{resp}"
           elif method == "blank":
               qtemp = "{event} {rel-natural} {resp}"
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
               anstemp = "{ph} {rel-natural}"
           elif method == "unsup-rel":
               qtemp = "{event} {ph} {resp}"
               anstemp = "{ph} {prefix}"
           elif method == "unsup-wrap-rel":
               qtemp = "{com_i} {event} {ph} {resp}"
               anstemp = "{prefix}"
           elif method == "rel-mask-wrap":
               qtemp = "{enc_com_token} {event} {ph} {resp}"
               anstemp = "{ph} {rel-natural}"
           elif method == "rel-unsup":
               qtemp = "{event} {rel-natural} {ph} "
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
               qtemp = "{rel_i} {event} {rel-natural} {gen} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-nat-tokens":
               qtemp = "{event} {nat-tokens}" 
               anstemp = "{resp} {end}"
           elif method == "unsup-nat-tokens":
               qtemp = "{event} {nat-tokens} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "sup-nat":
               #qtemp = "{rel-natural-pure}" 
               qtemp = "{rel-natural}"
               anstemp = "{resp} {end}"
           elif method == "sup-nat-tail":
               #qtemp = "{rel-natural-pure}" 
               qtemp = "{rel-natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-nat":
               qtemp = "{rel-natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-nat-head":
               qtemp = "{rel-natural}"
               anstemp = "{resp} {end}"
           elif method == "unsup-nat-n":
               qtemp = "{rel-nat-n}"
               anstemp = "{ph} {resp} {end}"
           elif method == "enc-unsup-nat":
               qtemp = "{rel_i} {event} {rel-natural} {ph}" 
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-nat-fa":
               qtemp = "{event} {rel_natural_fa} {ph}" 
               anstemp = "{ph} {resp} {end}"
          # uuuuuuuuuuuuuuuu
           elif method == "unsup-wrap-nat":
               #qtemp = "{rel_i} {rel-natural} {ph}"
               if tn == 1:
                   qtemp = "{c@lstm_6} {rel-natural}"
               elif tn == 2:
                   qtemp = "{c@merge_i} {rel-natural} "
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-nat-end":
               qtemp = "{rel-natural} {rel_i}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-nat-mid":
               qtemp = "{event} {rel_i} {rel-natural} {rel_i} {ph}" 
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
               qtemp = "{rel_i} {rel-natural}"
               anstemp = "{resp} {end}"
           elif method == "sup-no-gen":
               qtemp = "{event}"
               anstemp = "{resp} {end}"
           elif method == "gen":
               qtemp = "{gen}"
               anstemp = "{resp}"
           elif method == "pred-enfa":
               qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {ph} {event} {rel-natural} {rel_i_end} {gen_end} <extra_id_1>"
               anstemp = "{ph} {target_text} <extra_id_1> {resp} <extra_id_2>"
           elif method == "context-en":
               qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_en} {target_text} {rel_i_start} {event} {rel-natural} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "context-faen":
               qtemp = "{rel_i_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_en} {target_text} {event} {rel-natural} {rel_i_end} {gen_end} {ph}"
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
               qtemp = "{rel_i} {event} {rel-natural} {ph}"
               context = "{xAttr} {xIntent} {xReact}"
               ex_qtemp = "{rel_enc_token} {rel_natural_en} {target_text} {end}"
               anstemp = "{ph} {resp} {end}"
           elif method == "event-resp-n-wrap":
               qtemp = "{event} {examples} {rel_i} {event} {rel-natural} {ph}"
               ex_qtemp = "{rel_natural_en} {target_text} {end} \n"
               anstemp = "{ph} {resp} {end}"
           elif method == "gpt-event-resp-n-wrap":
               qtemp = "{examples} {rel_i}"
               ex_qtemp = "{rel_i} {input_text} {rel_natural_en} {target_text} {sep} \n"
               anstemp = "{event} {rel-natural} {resp} {end}"
           elif method == "fa-gpt-event-resp-n-wrap":
               qtemp = "{examples} {rel_i}"
               ex_qtemp = "{rel_i} {input_text_fa} {rel_natural_fa} {target_text_fa} {sep} \n"
               anstemp = "{event} {rel-natural} {resp} {end}"
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
               ex_qtemp = "{input_text} {rel-natural} {target_text}. {end} \n"
               anstemp = "{ph} {resp} {end}"
           elif method == "gpt-n-example":
               qtemp = "{examples} {event} {rel-natural}"
               ex_qtemp = "{input_text} {rel-natural} {target_text} {end}"
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
               qtemp = "{rel_i_start} {gen_start} {input_text} {rel_natural_en} {gen_fa} {target_text_fa} {rel_i_start} {event} {rel-natural} {rel_i_end} {gen_end} {ph}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-context-fa":
               qtemp = "{rel_i_start} {gen_start} {input_text_fa} {rel_natural_fa} {gen_fa} {target_text_fa} {rel_i_start} {event} {rel-natural} {rel_i_end} {gen_end} {ph}"
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
               qtemp = "{event} {rel-natural}"
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
               qtemp = "{tokens} {rel-natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-rel-nat":
               qtemp = "{prefix} {rel-natural}"
               anstemp = "{ph} {resp} {end}"
           elif method == "unsup-wrap-rel-ans":
               qtemp = "{event} {tokens} {ph} "
               anstemp = "{ph} {rel-natural-pure} {resp}"
           elif method == "unsup-nat-rel-ans":
               qtemp = "{event}. {ph}"
               anstemp = "{ph} {rel-natural-pure} {resp}."
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
           flags = self.get_flags(method)
           return qtemp, anstemp, ex_qtemp, ex_anstemp, flags

    def get_flags(self, method):
           flags = {
                   "method":method,
                   "wrap":"wrap" in method,
                   "freeze":"wrap" in method,
                   "unfreeze":False,
                   }
           return flags
           
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
                   q = q.replace("{rel-natural}","{rel-natural-pure}")
               q = fix_pos(q, gen_pos, prompt_pos)
               ret_q.append(q)
           qtemp = ret_q
           return qtemp, anstemp, ex_qtemp, ex_anstemp, context, flags

    def fill_consts(self, template, ex_temp, context,rel, row, rows=None, mask=-1, method="", someone=False):
        #dlog.info("fill consts, input text: %s", template)
        text = self.fill_const_for_rel(template, row)
        #dlog.info("fill consts, input text: %s", text)
        plen = relation_prompt_lengths[rel]
        text = text.replace("{ph}", placeholder_token)
        #dlog.info("fill consts, input text: %s", context)
        counter = 0
        pi = 0
        text = self.fill_prompt(text, rel, "{rel_i}")
        #text = self.fill_prompt_regex(text, rel, "{rel-([@a-zA-Z]+)_(\d+)}")
        text = self.fill_prompt_regex(text, rel, "{([@a-zA-Z]+)_(\d+)}")
        text = self.fill_prompt_regex(text, rel, "{([@a-zA-Z]+)_([a-zA-Z]+)}")
        text = self.fill_prompt(text, rel, "{tokens}")
        text = self.fill_prompt(text, rel, "{tokens-rand}")
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
                example = self.fill_const_for_rel(example, _row)
                example = self.fill_prompt(example, rel, "{rel_i}")
                example = self.fill_prompt(example, rel, "{tokens}")
                example = self.fill_prompt(example, rel, "{tokens-rand}")
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
                example = self.fill_const_for_rel(example, _row)
                example = self.fill_prompt(example, relation, "{rel_i}")
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


    def fill_vars(self, template, rel, event, resp, gen_token= "gen_en",  inp_lang="en", resp_lang="en", ph_num=3, someone=False):
        temp_num = self.temp_num
        if type(temp_num) == str and temp_num.isnumeric():
            temp_num = int(temp_num)
        event1, event2= "",""
        if "{@@@}" in event:
            #mbp("b")
            event1, event2 = event.split("{@@@}")
            event = event.replace("{@@@}"," ")
        if temp_num in rel_nat_maps[rel]:
            rel_natural = rel_nat_maps[rel][temp_num]        
        else:
            rel_natural = rel_nat_maps[rel][1]        
        rel_natural_tokens = rel_nat_maps[rel]["nat-tokens"]        
        rel_natural_pure = rel_natural.replace("{ph}", "")
        rel_natural_pure = rel_natural_pure.replace(".", "")
        rel_n = ""
        for i in range(ph_num):
            rel_n += "<extra_id_" + str(i) + "> "
        rel_nat_n = rel_natural.replace("{ph}", rel_n)
        rel_natural = rel_natural.replace("{ph}", placeholder_token)

        rep1  = {
                "{rel-natural}":rel_natural,
                "{rel-natural-pure}":rel_natural_pure,
                "{rel-nat-n}":rel_nat_n,
                "{nat-tokens}":rel_natural_tokens,
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
        text = self.fill_prompt(text, rel, "{rel_lang_i}", lang=lang)
        text = self.fill_prompt(text, rel, "{gen_lang}", lang=lang)
        return text


    def fill_const_for_rel(self, template, row):
        text = template
        if row is None:
            return text
        #dlog.debug("fill const for: %s", text)
        rel = row["prefix"]
        rel_token = rel_maps[rel]        
        rel_natural_en_postfix = rel_nat_maps[rel][1]        
        rel_natural_en_prefix = rel_nat_maps[rel][2]        
        rel_natural_fa = rel_nat_maps[rel]["fa"]        
        rep  = {"{rel}":rel, 
                "{rel_token}":rel_token,
                "{rel_natural_en}":rel_natural_en_postfix,
                "{rel_natural_en_pre}":rel_natural_en_prefix,
                "{rel_natural_fa}":rel_natural_fa,
                "{gen_fa}":gen_token_fa,
                "{sep}":sep,
                "{gen_en}":gen_token_en,
                "{end}":end_token}
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], template)
        for key,value in row.items():
            val = str(value)
            text = text.replace("{" + key + "}", val)
        return text

    def fill_prompt_regex(self, text, row_rel, regex):
        m = re.search(regex, text)
        while m: 
            if len(m.groups()) == 2:
                rel = m.groups()[0]
                emb = m.groups()[1]
                plen = "1"
                if emb.isdigit():
                    plen = emb
                num_holder = "_" + str(plen)
                if emb == "i":
                    plen = 0
                    num_holder = "_" + emb
                place_holder = "{" + rel + "_" + emb + "}"
            elif len(m.groups()) == 3:
                rel = m.groups()[0]
                emb = m.groups()[1]
                plen = m.groups()[2]
                num_holder = "_" + plen
                place_holder = "{" + rel + "_" + emb + "_" + plen + "}"
            if plen != 0:
                plen = [int(plen)]
            if rel == "rel":
                rel = row_rel
            text = self.fill_prompt(text, rel, place_holder, plen=plen, 
                    num_holder=num_holder, row_rel=row_rel)
            m = re.search(regex, text)
        return text

    def fill_prompt(self, text, rel, place_holder, counter = 0, lang="", plen = 0, num_holder="_i", row_rel=""):
        if not row_rel: row_rel = rel
        pi = 0
        if plen==0: 
            if rel in relation_prompt_lengths:
                plen = relation_prompt_lengths[rel]
            else:
                if False: #"merge" in rel or "mat" in rel:
                    plen = [self.num_prompts]  
                else:
                    plen = [self.num_prompt_tokens]  
        _pholder = place_holder
        place_holder = place_holder.replace("{", "<")  
        place_holder = place_holder.replace("}", ">")  
        place_holder = place_holder.replace("rel", row_rel)  
        place_holder = place_holder.replace("lang", lang)  
        #dlog.info("text: %s", text)
        while _pholder in text:
            if num_holder in _pholder:
                enc_plen = plen[pi] if pi < len(plen) else plen[-1] 
                prompt = ""
                for i in range(counter, counter + enc_plen):
                    token = place_holder
                    if num_holder != "_1":
                        token = token.replace(num_holder, "_" + str(i))  
                    else:
                        token = token.replace(num_holder, "")  
                    prompt += " " + token
            elif _pholder == "{tokens}": 
                prompt = rel_nat_maps[rel]["tokens"]
            elif _pholder == "{tokens-rand}": 
                permute = rel_nat_maps[rel]["tokens"].split()
                random.shuffle(permute)
                prompt = " ".join(permute)
            else:
                #mlog.info("************** using tokens of pholder %s",_pholder)
                prompt = place_holder
            prompt = prompt.strip()
            enc_plen = len(prompt.split())
            for token in prompt.split():
                if not rel in self.encoder_prompts:
                    self.encoder_prompts[rel] = []
                if not token in self.encoder_prompts[rel]:
                    self.encoder_prompts[rel].append(token)
            text = text.replace(_pholder,prompt, 1)
            counter += enc_plen 
            pi += 1
        #dlog.info("text: %s", text)
        return text

