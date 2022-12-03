
from comet.train.common import *
from comet.train.template import *
import pandas as pd
from datetime import datetime
from pathlib import Path
import random


class xAttrTemplate(RelTemplate):
    def get_templates(self, method, index, **kwargs):
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event} {ph}"
       anstemp = "{ph} {resp} {end}"
       flags = self.get_flags(method)
       tn = self.temp_num
       if tn.isdigit():
           tn = int(tn)
       if method == "unsup-wrap-nat":
           if tn == 1:
               qtemp = "{c@mat_i} {event}, So PersonX is seen as {ph}."
           elif tn == "merge":
               qtemp = "{c@merge_i} {event}, {rel_i} So PersonX is seen as {ph}."
           elif tn == "mat":
               qtemp = "{c@mat_i} {event}, {rel_i} So PersonX is seen as {ph}."
           elif tn == 3:
               qtemp = "{c@merge_i} {event}, So PersonX is seen as {ph}."
           elif tn == 4:
               qtemp = "{c@lstm_i} {event}, {rel_i} So PersonX is seen as {ph}."
           elif tn == 5:
               qtemp = "{rel@lstm_i} {event}, {rel_i} So PersonX is seen as {ph}."
           elif tn == 6:
               qtemp = "{event}, {emp_so_1} {emp_they_1} are {emb_seen_1} as {ph}."
           elif tn == 641:
               qtemp = "{c_6} {rel_4} {event}, they seen as {ph}"
           elif tn == 6411:
               qtemp = "{c_6} {event}, {rel_4} {ph}"
           elif tn == 6412:
               qtemp = "{d_6} {event}, {rel_4} {ph}"
           elif tn == 643:
               qtemp = "{event}, {c_6} {rel_4} they seen as {ph}"
           elif tn == 6431:
               qtemp = "{rel_4} {event}, {c_6} {ph}"
           elif tn == 642:
               qtemp = "{da_6} {c_4} {event}, {b_they} seen as {ph}"
           elif tn == 65:
               qtemp = "{because_1} {of_1} {event}, {they_1} {seen_1} as {ph}"
           elif tn == 66:
               qtemp = "{because_1} {of_1} {event}, {they_1} {seen_1} as {resp}"
               anstemp = "{rel_1}"
       else:
          return super().get_templates(method, index, **kwargs)
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

class xIntentTemplate(RelTemplate):
    def get_templates(self, method, index, **kwargs):
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event}"
       anstemp = "{ph} {resp} {end}"
       flags = self.get_flags(method)
       tn = self.temp_num
       if tn.isdigit():
           tn = int(tn)
       if method == "unsup-nat":
           if tn == 1:
               qtemp = "Why does {event}? {ph}"
               anstemp = "{ph} Because he wants {resp} {end}."
           elif tn == 2:
               qtemp = "Why does {event}? Because he wants {ph}"
               anstemp = "{ph} {resp} {end}"
           elif tn == 3:
               qtemp = "Why does {event}? Because {ph}"
               anstemp = "he intends {ph} {resp} {end}"
       elif method == "sup-nat":
           if tn == 1:
               qtemp = "Why does {event}?"
               anstemp = "Because he wants {resp} {end}"
           elif tn == 2:
               qtemp = "Why does {event}? Because he wants"
               anstemp = "{resp} {end}"
           elif tn == 3:
               qtemp = "Why does {event}? Because "
               anstemp = "he intends {resp} {end}"
       elif method == "unsup-wrap-nat":
           if tn == 1:
               qtemp = "{c@mat_i} Because of {event}, they want {ph}"
           if tn == "merge":
               qtemp = "{c@merge_i} Because of {event}, {rel_i} they want {ph}"
           if tn == "mat":
               qtemp = "{c@mat_i} Because of {event}, {rel_i} they want {ph}"
           if tn == 3:
               qtemp = "{c@merge_i} Because of {event}, they want {ph}"
           if tn == 4:
               qtemp = "{c@lstm_i} Because of {event}, {rel_i} they want {ph}"
           if tn == 5:
               qtemp = "{rel@lstm_i} Because of {event}, {rel_i} they want {ph}"
           elif tn == 6:
               qtemp = "{emb_because_1} {emb_of_1} {event}, {emb_they_1} {emb_want_1} {ph}"
           elif tn == 61:
               qtemp = "{because_1} {of_1} {event}, {they_1} {want_1} {ph}"
           elif tn == 62:
               qtemp = "{rel_4} {event}, {ph}"
           elif tn == 63:
               qtemp = "{rel_5} Because of {event}, they want {ph}"
           elif tn == 641:
               qtemp = "{c_6} {rel_4} {event}, they want {ph}"
           elif tn == 6411:
               qtemp = "{c_6} {event}, {rel_4}  {ph}"
           elif tn == 6412:
               qtemp = "{e_6} {event}, {rel_4}  {ph}"
           elif tn == 643:
               qtemp = "{event}, {c_6} {rel_4} they want {ph}"
           elif tn == 6431:
               qtemp = "{rel_4} {event}, {c_6} {ph}"
           elif tn == 642:
               qtemp = "{db_6} {c_4} {event}, {b_they} want {ph}"
           elif tn == 652:
               qtemp = "{because_1} {of_1} {event}, {rel_4} want {ph}"
           elif tn == 653:
               qtemp = "{a_because} {a_of} {event}, {rel_4} want {ph}"
           elif tn == 66:
               qtemp = "{because_1} {of_1} {event}, {they_1} {want_1} {resp}"
               anstemp = "{rel_1}"
       else:
          return super().get_templates(method, index, **kwargs)
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

class CBTemplate(RelTemplate):
    def get_templates(self, method, index, **kwargs):
       if self.split_name != "train" and method == "mix":
           method = "unsup-wrap"
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event}"
       anstemp = "{resp}"
       methods = ["unsup","unsup-wrap","sup"]
       mti = self.method_index % len(methods)
       if index % self.batch_size == 0:
           self.method_index += 1
       wrap = False
       freeze = False
       unfreeze = True
       if method == "mix":
          method = methods[mti]
       if method == "unsup":
          qtemp =  "{event2}. {event1} ? {ph}"
          anstemp = "{ph} {resp} {end}"
       elif method == "sup":
          qtemp =  "{event2}. {event1}? "
          anstemp = "{resp} {end}"
       elif method == "unsup-wrap":
          qtemp =  "{event2}. {cibi_4} {rel_3} {event1} ? {ph}"
          anstemp = "{ph} {resp} {end}"
          wrap = True
          freeze = True
          unfreeze = False
       else: 
          return super().get_templates(method, index, **kwargs)
       flags = {
               "method":method,
               "wrap": wrap,
               "freeze": freeze,
               "unfreeze": unfreeze
               }
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

template_conf = {
        #"cb": CBTemplate,
        "xIntent":xIntentTemplate,
        "xAttr":xAttrTemplate, 
        }
