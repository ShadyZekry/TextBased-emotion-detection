verb_tags = ["VB", "VBD", "VBN", "VBP", "VBZ"]
noun_tags = ["NN", "NNS"]


def extract_verb_noun(sentence_trees):
    phrases = []
    for subtree in sentence_trees[0].subtrees():
        if subtree.label() in noun_tags:
            t = subtree
            t = ' '.join(t.leaves())
            phrases.append(t)
        if subtree.label() in verb_tags:
            t = subtree
            t = ' '.join(t.leaves())
            phrases.insert(0, t)
    return phrases
