import logging
import os
from os.path import expanduser
from pytz import timezone
import datetime
from pathlib import Path

main_args = {}
def set_args(args):
    global main_args 
    main_args =args

def myargs(key):
    return main_args[key]

def tag():
    tag = main_args["tag"]
    _tag = ""
    for _t in tag.split("@"):
        if _t in main_args:
            _tag += "|" + _t  + "=" + str(main_args[_t])
        else:
            _tag += "|" + _t  
    return _tag.strip("|")

tehran = timezone('Asia/Tehran')
now = datetime.datetime.now(tehran)
now = now.strftime('%Y-%m-%d-%H:%M')
home = expanduser("~")
colab = not "ahmad" in home and not "pouramini" in home
if not colab: 
    logPath = os.path.join(home, "logs")
    resPath = os.path.join(home, "results") 
    pretPath = os.path.join(home, "pret") 
else:
    home = "/content/drive/MyDrive/pouramini"
    pretPath = "/content/drive/MyDrive/pret"
    logPath = "/content/drive/MyDrive/logs"
    resPath = "/content/drive/MyDrive/logs/results"

pp = Path(__file__).parent.parent.resolve()
dataPath = os.path.join(pp, "data", "atomic2020")
confPath = "base_confs" 

Path(resPath).mkdir(exist_ok=True, parents=True)
Path(logPath).mkdir(exist_ok=True, parents=True)

logFilename = os.path.join(logPath, "all.log") #app_path + '/log_file.log'
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")
FORMAT2 = logging.Formatter("%(message)s")
logging.basicConfig(filename=logFilename)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(FORMAT2)
mlog = logging.getLogger("comet.main")
mlog.setLevel(logging.INFO)
mlog.addHandler(consoleHandler)
clog = logging.getLogger("comet.cfg")
dlog = logging.getLogger("comet.data")
vlog = logging.getLogger("comet.eval")
tlog = logging.getLogger("comet.train")
timelog = logging.getLogger("comet.time")

import inspect
import sys
STOP_LEVEL = 0
def mbp(sl=-1):
    if colab: return
    if sl == STOP_LEVEL or sl == 0: 
        fname = sys._getframe().f_back.f_code.co_name
        line = sys._getframe().f_back.f_lineno
        mlog.info("break point at %s line %s",fname, line)
        #breakpoint()

def trace(frame, event, arg):
    if event == "call":
        filename = frame.f_code.co_filename
        if filename.endswith("train/train.py"):
            lineno = frame.f_lineno
            # Here I'm printing the file and line number,
            # but you can examine the frame, locals, etc too.
            print("%s @ %s" % (filename, lineno))
    return trace

mlog.info(now)
#sys.settrace(trace)

for logger, fname in zip([mlog,dlog,clog,vlog,tlog,timelog], ["all_main","all_data","all_cfg","all_eval","all_train", "all_time"]):
    logger.setLevel(logging.INFO)
    logFilename = os.path.join(logPath, fname + ".log")
    handler = logging.FileHandler(logFilename, mode="w")
    handler.setFormatter(FORMAT)
    logger.addHandler(handler)

