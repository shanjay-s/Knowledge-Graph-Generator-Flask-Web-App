import pandas as pd
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import re
import spacy
import neuralcoref
import networkx as nx
import matplotlib.pyplot as plt
import os
from flask import session
import random
import string

nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)
from flask import Flask,render_template,request,url_for

FOLDER = os.path.join('static', 'knowledge-graphs')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FOLDER


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["GET","POST"])
def predict():
    def get_entity_pairs(text, coref=True):
        # preprocess text
        text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
        text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
        text = nlp(text)
        if coref:
            text = nlp(text._.coref_resolved)  # resolve coreference clusters

        def refine_ent(ent, sent):
            unwanted_tokens = (
                'PRON',  # pronouns
                'PART',  # particle
                'DET',  # determiner
                'SCONJ',  # subordinating conjunction
                'PUNCT',  # punctuation
                'SYM',  # symbol
                'X',  # other
            )
            ent_type = ent.ent_type_  # get entity type
            if ent_type == '':
                ent_type = 'NOUN_CHUNK'
                ent = ' '.join(str(t.text) for t in
                               nlp(str(ent)) if t.pos_
                               not in unwanted_tokens and t.is_stop == False)
            elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
                refined = ''
                for i in range(len(sent) - ent.i):
                    if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                        refined += ' ' + str(ent.nbor(i))
                    else:
                        ent = refined.strip()
                        break

            return ent, ent_type

        sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
        ent_pairs = []
        for sent in sentences:
            sent = nlp(sent)
            spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
            spans = spacy.util.filter_spans(spans)
            with sent.retokenize() as retokenizer:
                [retokenizer.merge(span, attrs={'tag': span.root.tag,'dep': span.root.dep}) for span in spans]
            deps = [token.dep_ for token in sent]

            # limit our example to simple sentences with one subject and object
            if (deps.count('obj') + deps.count('dobj')) != 1 or (deps.count('subj') + deps.count('nsubj')) != 1:
                continue

            for token in sent:
                if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                    continue
                subject = [w for w in token.head.lefts if w.dep_ in ('subj', 'nsubj')]  # identify subject nodes
                if subject:
                    subject = subject[0]
                    # identify relationship by root dependency
                    relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                    if relation:
                        relation = relation[0]
                        # add adposition or particle to relationship
                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join((str(relation), str(relation.nbor(1))))
                    else:
                        relation = 'unknown'

                    subject, subject_type = refine_ent(subject, sent)
                    token, object_type = refine_ent(token, sent)

                    ent_pairs.append([str(subject), str(relation), str(token),str(subject_type), str(object_type)])

        ent_pairs = [sublist for sublist in ent_pairs if not any(str(ent) == '' for ent in sublist)]
        pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object', 'subject_type', 'object_type'])
        print('Entity pairs extracted:', str(len(ent_pairs)))

        return pairs

    word = request.form['keyword']
    pairs = get_entity_pairs(word)

    def draw_kg(pairs):
        k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object', create_using=nx.MultiDiGraph())
        node_deg = nx.degree(k_graph)
        layout = nx.spring_layout(k_graph, k=0.5, iterations=20)
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        nx.draw_networkx(
            k_graph,
            node_size=[int(deg[1]) * 500 for deg in node_deg],
            arrowsize=5,
            linewidths=1.5,
            pos=layout,
            edge_color='red',
            edgecolors='black',
            node_color='skyblue',
        )
        char_set = string.ascii_uppercase + string.digits
        file_str = ''.join(random.sample(char_set * 6, 6)) + '.png'
        labels = dict(zip(list(zip(pairs.subject, pairs.object)), pairs['relation'].tolist()))
        nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,font_color='red')
        plt.axis('off')
        file_str_fi = 'static/knowledge-graphs/' + file_str
        plt.savefig(file_str_fi)
        plt.close()
        return file_str
    st = draw_kg(pairs)
    full_file = os.path.join(app.config['UPLOAD_FOLDER'], st)
    return render_template("result.html", user_image = full_file)


@app.route("/back",methods=["GET","POST"])
def back():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)

