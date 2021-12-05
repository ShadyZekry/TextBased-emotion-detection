import os
from nltk import parse
from nltk.parse import stanford
from nltk.tree import Tree

# Download this file https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip 
# Extract "stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar" and "stanford-parser-full-2020-11-17\stanford-parser.jar" in a new folder called "jar".

# Download https://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar
# Open it with Winrar, Extract "edu\stanford\nlp\models\lexparser\englishPCFG.caseless.ser.gz" in the same project directory.

def parse_sentences(input):
	projectPath = 'E:\\College\\GP\\TextBased-emotion-detection\\social_engineering\\'
	os.environ['STANFORD_PARSER'] = projectPath+'jars\\stanford-parser.jar'
	os.environ['STANFORD_MODELS'] = projectPath+'jars\\stanford-parser-4.2.0-models.jar'
	os.environ['JAVAHOME'] = 'C:\\Program Files\\AdoptOpenJDK\\jdk-15.0.1.9-hotspot\\bin\\java.exe'

	parser = stanford.StanfordParser(model_path=projectPath+"englishPCFG.caseless.ser.gz")
	return parser.raw_parse_sents(input)