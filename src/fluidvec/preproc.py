from itertools import chain

def serialize_components(ch, ctree):
    try:
        compos = ctree.query(ch, use_flag="shortest", max_depth=1)[0]
    except Exception as ex:
        compos = ""
    if isinstance(compos, str):
        return ["<COMPO_NA>"]
    
    idc = compos.idc
    serialized = []
    for i, c in enumerate(compos.components()):                   
        serialized.append(f"{idc}{i}-{str(c)}")        
    return serialized

def make_charpos(word):
    charpos = []
    for i in range(len(word)):
        cp = ""
        if i == 0:
            cp += "_"
        cp += word[i]
        if i == len(word)-1:
            cp += "_"
        charpos.append(cp)
    return charpos
    
def make_word_tuple(word, ctree):
    """Construct a tuple representing word informations

    Parameters
    -----------
    word: string
          the word to make tuple with
    ctree: ComponentTree
           the ComponentTree instance for constructing components of characters

    Returns
    --------
    (compos, chars, word)
           A tuple consisting of components, list of strings, 
           chars, list of strings, and word, a string.
    """
    chars = make_charpos(word)
    compos = [serialize_components(ch, ctree) for ch in word]
    compos = list(chain.from_iterable(compos))
    return (compos, chars, word)