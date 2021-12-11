def is_not_question(trees):
    phrases = []
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == '.' and subtree[0]=='?':
                return False
    return True

# S     => ordinary sentence
# SQ    => Yes or No question
# SBARQ => Open questions