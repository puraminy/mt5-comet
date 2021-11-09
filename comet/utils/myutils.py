def superitems(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            for i in superitems(v):
                yield (k,) + i
    else:
        yield (obj,)

def dictPath(path, dictionary, val, sep="_"):
    "set a value in a nested dictionary"
    while path.startswith(sep):
        path = path[1:]
    parts = path.split(sep, 1)
    if len(parts) > 1:
        branch = dictionary.setdefault(parts[0], {})
        dictPath(parts[1], branch, val, sep)
    else:
        dictionary[parts[0]] = val

def getVal(path, dictionary, sep="_"):
    "get a value in a nested dictionary"
    while path.startswith(sep):
        path = path[1:]
    parts = path.split(sep, 1)
    if len(parts) > 1:
        branch = dictionary.setdefault(parts[0], {})
        return getVal(parts[1], branch, sep)
    else:
        if parts[0] in dictionary:
            return dictionary[parts[0]]
        else:
            return "NA"

def arg2dict(arg):
    # converts a string like opt1:val, opt2:val to a dictionary
    return dict(
        map(str.strip, sub.split(":", 1))
        for sub in arg.split(",")
        if ":" in sub
    )


def to_unicode(text, remove_quotes=True):
    if remove_quotes:
        text = text[2:-1]
    text = text.encode("raw_unicode_escape")
    text = text.decode("unicode_escape")
    text = text.encode("raw_unicode_escape")
    return text.decode()
