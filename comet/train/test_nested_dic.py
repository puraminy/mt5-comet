results = {}
import json
from comet.utils.myutils import *
def main():
    mean_rouge = {"en2en":3, "fa2en":4}
    mean_bert = {"en2en":6, "fa2en":7}
    val = {"rouge":mean_rouge, "bert":mean_bert}
    dictPath("A-C_Val_char", results, val, sep="_")
    res_str = json.dumps(results, indent=2)
    print(res_str)

if __name__ == "__main__":
    main()
