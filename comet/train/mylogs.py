import logging
import os
from os.path import expanduser
from pytz import timezone
import datetime
from pathlib import Path

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
    logPath = "/content/"
    resPath = "/content/drive/MyDrive/pouramini/results"

Path(resPath).mkdir(exist_ok=True, parents=True)
Path(logPath).mkdir(exist_ok=True, parents=True)

logFilename = os.path.join(logPath, "all.log") #app_path + '/log_file.log'
FORMAT = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s")
logging.basicConfig(filename=logFilename)
consoleHandler = logging.StreamHandler()
mlog = logging.getLogger("comet.main")
mlog.setLevel(logging.INFO)
mlog.addHandler(consoleHandler)
clog = logging.getLogger("comet.cfg")
dlog = logging.getLogger("comet.data")
vlog = logging.getLogger("comet.eval")
tlog = logging.getLogger("comet.train")
timelog = logging.getLogger("comet.time")


mlog.info(now)

for logger, fname in zip([mlog,dlog,clog,vlog,tlog,timelog], ["all_main","all_data","all_cfg","all_eval","all_train", "all_time"]):
    logger.setLevel(logging.INFO)
    logFilename = os.path.join(logPath, fname + ".log")
    handler = logging.FileHandler(logFilename, mode="w")
    handler.setFormatter(FORMAT)
    logger.addHandler(handler)

