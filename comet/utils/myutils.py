
def dictPath(string, dictionary, val, sep="/"):
    "set value in a nested dictionaries"
    while string.startswith(sep):
        string = string[1:]
    parts = string.split(sep, 1)
    if len(parts) > 1:
        branch = dictionary.setdefault(parts[0], {})
        dictPath(parts[1], branch, val)
    else:
        dictionary[parts[0]] = val

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
