import re

element = "{event_en} this is {event_fa}"
z = re.findall("\{event_(\w+)\}", element)
for x in z:
    print(x)

