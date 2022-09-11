
from comet.train.common import *
from comet.train.dataset import *
import pandas as pd
from datetime import datetime
from pathlib import Path
import random


class xAttrDataset(MyDataset):
    def get_templates(self, method, index, **kwargs):
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event} {ph}"
       anstemp = "{ph} {resp} {end}"
       flags = self.get_flags(method)
       if method == "unsup-wrap-nat":
           qtemp = "{rel_8} {event}, So PersonX is seen as {ph}."
       else:
          return super().get_templates(method, **kwargs)
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

class xIntentDataset(MyDataset):
    def get_templates(self, method, index, **kwargs):
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event}"
       anstemp = "{ph} {resp} {end}"
       flags = self.get_flags(method)
       if method == "unsup-nat":
           if int(self.temp_num) == 1:
               qtemp = "Why does {event}? {ph}"
               anstemp = "{ph} Because he wants {resp} {end}."
           elif int(self.temp_num) == 2:
               qtemp = "Why does {event}? Because he wants {ph}"
               anstemp = "{ph} {resp} {end}"
       elif method == "sup-nat":
           qtemp = "Why does {event}?"
           anstemp = "Because he wants {resp} {end}"
       elif method == "unsup-wrap-nat":
           qtemp = "{emb_because_1} {emb_test_2} Because of {event}, they want {ph}"
       else:
          return super().get_templates(method, **kwargs)
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

class CBDataset(MyDataset):
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
          return super().get_templates(method, **kwargs)
       flags = {
               "method":method,
               "wrap": wrap,
               "freeze": freeze,
               "unfreeze": unfreeze
               }
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

data_conf = {
        "cb": CBDataset,
        "xIntent":xIntentDataset,
        "xAttr":xAttrDataset 
        }
