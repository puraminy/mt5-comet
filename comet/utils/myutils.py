import click
def call_click_command(cmd, *args, **kwargs):
    """ Wrapper to call a click command

    :param cmd: click cli command function to call 
    :param args: arguments to pass to the function 
    :param kwargs: keywrod arguments to pass to the function 
    :return: None 
    """

    # Get positional arguments from args
    arg_values = {c.name: a for a, c in zip(args, cmd.params)}
    args_needed = {c.name: c for c in cmd.params
                   if c.name not in arg_values}

    # build and check opts list from kwargs
    opts = {a.name: a for a in cmd.params if isinstance(a, click.Option)}
    for name in kwargs:
        if name in opts:
            arg_values[name] = kwargs[name]
        else:
            if name in args_needed:
                arg_values[name] = kwargs[name]
                del args_needed[name]
            else:
                raise click.BadParameter(
                    "Unknown keyword argument '{}'".format(name))


    # check positional arguments list
    for arg in (a for a in cmd.params if isinstance(a, click.Argument)):
        if arg.name not in arg_values:
            raise click.BadParameter("Missing required positional"
                                     "parameter '{}'".format(arg.name))

    # build parameter lists
    opts_list = sum(
        [[o.opts[0], str(arg_values[n])] for n, o in opts.items()], [])
    args_list = [str(v) for n, v in arg_values.items() if n not in opts]

    # call the command
    cmd(opts_list + args_list)


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
