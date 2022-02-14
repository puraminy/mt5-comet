
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9]+)"

def rplit_into_sentences(text):
    sents = nltk.sent_tokenize(text)
    return sents

def qplit_into_sentences(text):
    try:
        import nltk
        try:
            sents = nltk.sent_tokenize(text)
            return sents
        except LookupError:
            nltk.download('punkt')
            sents = nltk.sent_tokenize(text)
            return sents
    except ImportError as e:
        return rplit_into_sentences(text)

def split_into_sentences(text, debug = False, limit = 15, split_on = ['.','?','!',':']):
    text = " " + text + "  "

    rep = {"Ph.D.": "Ph<prd>D<prd>",
            "[FRAG]":"<stop>",
            "et al.":"et al<prd>",
            " et.":" et<prd>",
            "e.g.":"e<prd>g<prd>",
            "e.g":"e<prd>g",
            "vs.":"vs<prd>",
            "www.":"www<prd>",
            "etc.":"etc<prd>",
            "i.e.":"i<prd>e<prd>",
            "...":"<prd><prd><prd>",
            "•":"<stop>•"
           }

    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    #print(text)

