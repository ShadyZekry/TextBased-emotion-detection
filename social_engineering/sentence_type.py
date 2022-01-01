closed_question_tag = 'SQ'
opened_question_tag = 'SBARQ'
sentence_tag = 'S'


def is_sentence(sentences_trees):
    if sentences_trees[0].label() == sentence_tag:
        return True
    return False

def is_closed_question(sentences_trees):
    if sentences_trees[0].label() == closed_question_tag:
        return True
    return False
