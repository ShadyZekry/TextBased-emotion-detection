import parse
import question
import sentence_type

sentences = parse.parse_sentences(("Hello, My name is Melroy.", "What is your name?","please, can you reset the router."))

for sentence in sentences:
    for tree in sentence:
        if(sentence_type.is_not_question(tree)):
            print(question.extract_verb_object(tree))