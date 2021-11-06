results = {}
import json
def dictPath(string, dictionary, val):
    while string.startswith('/'):
        string = string[1:]
    parts = string.split('/', 1)
    if len(parts) > 1:
        branch = dictionary.setdefault(parts[0], {})
        dictPath(parts[1], branch, val)
    else:
        dictionary[parts[0]] = val

def add(output_name):
    mean_rouge = {"en2en":3, "fa2en":4}
    mean_bert = {"en2en":6, "fa2en":7}
    output_name = output_name.replace("_","/")
    dpath.util.set(results, path, SOME_VALUE)
    res_str = json.dumps(results, indent=2)
    print(res_str)
    return

    res = results
    parts = output_name.split("/")
    for p in parts:
        res = res.setdefault(p, {})
    print(res)
    res["rouge"] = mean_rouge
    res["bert"] = mean_bert


def main():
    mean_rouge = {"en2en":3, "fa2en":4}
    mean_bert = {"en2en":6, "fa2en":7}
    val = {"rouge":mean_rouge, "bert":mean_bert}
    dictPath("A-C/Val/char", results, val)
    res_str = json.dumps(results, indent=2)
    print(res_str)

if __name__ == "__main__":
    main()
