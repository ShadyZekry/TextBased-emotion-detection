def extract_verb_object(trees):
    phrases = []
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == 'NN':
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)
            if subtree.label() in ['VB', 'VBZ']:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.insert(0, t)

    return phrases
