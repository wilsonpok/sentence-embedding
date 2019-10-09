import sys
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from pathlib import Path

home = str(Path.home())


# Inputs
input_file = home + sys.argv[1]
# input_file = home + '/sentence-embedding/csv/dummy-sentences-no-header.csv'

# Outputs
output_file = home + sys.argv[2]
# output_file = home + '/sentence-embedding/csv/dummy-embeddings-no-header.csv'


##########################
# Load data
##########################

df = pd.read_csv(input_file, header=None)

print(df.shape)
# (21, 2)

df.columns.tolist()
# [0, 1]

df = df.loc[df[1].notnull()]

print(df.shape)
# (21, 2)


##########################
# Create embeddings
##########################

# Import the Universal Sentence Encoder's TF Hub module
module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
embed = hub.Module(module_url)

# Compute a representation for each sentence
sentences = np.asarray(df[1])

# Reduce logging output
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  sentence_embeddings = session.run(embed(sentences))

  for i, sentence_embedding in enumerate(np.array(sentence_embeddings).tolist()):
    print("Message: {}".format(sentences[i]))
    print("Embedding size: {}".format(len(sentence_embedding)))
    sentence_embedding_snippet = ", ".join(
        (str(x) for x in sentence_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(sentence_embedding_snippet))

sentence_embeddings.shape
# (21, 512)



##############################
# Output embedding
##############################

embedding_df = pd.DataFrame(sentence_embeddings)

embedding_df.columns = ['x' + str(col + 1) for col in embedding_df.columns]

output_embeddings = pd.concat([df[0], embedding_df], axis=1)

output_embeddings.shape
#  (21, 514)

output_embeddings.to_csv(output_file, index=False)
