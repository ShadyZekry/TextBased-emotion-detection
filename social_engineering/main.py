import parse
import question
import sentence_type

sentences = parse.parse_sentences(("Do you get up early?","Hello, My name is Melroy.", "What is your name?","please, can you reset the router."))

for sentence in sentences:
    for tree in sentence:
        if(sentence_type.is_sentence(tree)):
            print(question.extract_verb_noun(tree))