
from comet.train.common import *
from comet.train.dataset import *
import pandas as pd
from datetime import datetime
from pathlib import Path
import random


class xAttrDataset(MyDataset):
    pass


class CBDataset(MyDataset):
    def get_templates(self, method, **kwargs):
       ex_qtemp = ""
       ex_anstemp = ""
       qtemp = "{event}"
       anstemp = "{resp}"
       methods = ["unsup","unsup-wrap","sup"]
       mbp("b")
       if method == "mix":
           method = random.choice(methods)
       if method == "unsup":
          qtemp =  "{event2}. {event1} ? {ph}"
          anstemp = "{ph} {resp} {end}"
       elif method == "sup":
          qtemp =  "{event2}. {event1}? "
          anstemp = "{ph} {resp} {end}"
       elif method == "unsup-wrap":
          qtemp =  "{event2}. {cibi_4} {rel_3} {event1} ? {ph}"
          anstemp = "{ph} {resp} {end}"
       else: 
          return super().get_templates(method, **kwargs)

       return qtemp, anstemp, ex_qtemp, ex_anstemp 

data_conf = {
        "cb": CBDataset
        }
