
from comet.train.common import *
from comet.train.dataset import *
import pandas as pd
from datetime import datetime
from pathlib import Path
import random


class xAttrDataset(MyDataset):
    pass


class CBDataset(MyDataset):
    def get_templates(self, method, index, **kwargs):
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event}"
       anstemp = "{resp}"
       methods = ["unsup","unsup-wrap","sup"]
       mbp("")
       if index % self.batch_size == 0:
           self.method_index += 1
       mti = self.method_index % len(methods)
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
       else: 
          return super().get_templates(method, **kwargs)
       flags = {
               "method":method,
               "wrap":"wrap" in method,
               "freeze":"wrap" in method,
               "unfreeze":not "wrap" in method,
               }
       return qtemp, anstemp, ex_qtemp, ex_anstemp, flags 

data_conf = {
        "cb": CBDataset
        }
