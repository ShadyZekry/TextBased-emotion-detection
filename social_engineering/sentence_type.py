import os
from nltk import parse
from nltk.parse import stanford
from nltk.tree import Tree

# Download this file https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip 
# Extract "stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar" and "stanford-parser-full-2020-11-17\stanford-parser.jar" in a new folder called "jar".

# Download https://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar
# Open it with Winrar, Extract "edu\stanford\nlp\models\lexparser\englishPCFG.caseless.ser.gz" in the same project directory.

projectPath = 'E:\\College\\GP\\TextBased-emotion-detection\\social_engineering\\'
os.environ['STANFORD_PARSER'] = projectPath+'jars\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = projectPath+'jars\\stanford-parser-4.2.0-models.jar'
os.environ['JAVAHOME'] = 'C:\\Program Files\\AdoptOpenJDK\\jdk-15.0.1.9-hotspot\\bin\\java.exe'

parser = stanford.StanfordParser(model_path=projectPath+"englishPCFG.caseless.ser.gz")
sentences = parser.raw_parse_sents(("Hello, My name is Melroy.", "What is your name?","please, can you reset the router."))


def is_not_question(trees):
    phrases = []
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == '.' and subtree[0]=='?':
                return False
    return True

def extract_verb_object(trees):
    phrases = []
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() =='NN' :
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)
            if subtree.label() in ['VB', 'VBZ']:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.insert(0,t)

    return phrases

for sentence in sentences:
    for tree in sentence:
        if(is_not_question(tree)):
            print(extract_verb_object(tree))