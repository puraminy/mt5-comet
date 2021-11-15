import re

res = re.search('\{event(\d+)\}', 'This is a generated number {event123} {resp123} which is an integer.')

if res is not None:
    integer = int(res.group(1))
    print(integer)


