import json
import os
import re
from warnings import warn
from shutil import rmtree
import tempfile
import zipfile

from elasticsearch_dsl import Mapping, field, Search
from elasticsearch.helpers import bulk, streaming_bulk
import numpy as np
import pandas as pd
import spacy
from sqlalchemy import inspect

op = os.path
POS = spacy.attrs.POS

PATTERNS = [
    ('np', [{POS: "ADJ", 'OP': '*'}, {POS: 'NOUN', 'OP': "+"}]),
    ('pnp', [{POS: "ADJ", "OP": "*"}, {POS: "PROPN", "OP": "+"}])
]

PD2ES_TYPES = {
    np.dtype('O'): field.Text(),
    np.dtype('int64'): field.Integer(),
    np.dtype('float64'): field.Double(),
    np.dtype('<M8[ns]'): field.Date(),
    np.dtype('bool'): field.Boolean()
}


def load_spacy_model():
    nlp = spacy.load('en_core_web_lg')
    nlp.tokenizer.infix_finditer = re.compile(r'[~\-_]').finditer
    return nlp


def make_matcher(nlp, patterns=PATTERNS):
    matcher = spacy.matcher.Matcher(nlp.vocab)
    for _id, pattern in PATTERNS:
        matcher.add(_id, None, pattern)
    return matcher


def is_overlap(x, y):
    """Whether the token x is contained within any span in the sequence y."""
    return any([x.text in yy for yy in y])


def unoverlap(tokens):
    """From a set of tokens, remove all tokens that are contained within
    others."""
    textmap = {c.text: c for c in tokens}
    text_tokens = textmap.keys()
    newtokens = []
    for token in text_tokens:
        if not is_overlap(textmap[token], text_tokens - {token}):
            newtokens.append(token)
    return [textmap[t] for t in newtokens]


def _find_es_types(df):
    return {c: PD2ES_TYPES.get(df[c].dtype) for c in df}


def preprocess_table(table, engine, nlp, headers_only=True, schema='masterdb'):
    """Preprocess a table for consumption by the search pipeline.

    Parameters
    ----------
    table: str
        Table name.
    engine: sqlalchemy.engine.Engine
    nlp : spacy.lang.en.English
        spaCy model
    headers_only: bool, optional
        If true, only table headers are read and processed (default), else all
        rows are also preprocessed.
    schema: str, optional
        Schema where the table resides.
    """
    # TODO: Also support dataframe inputs
    if not headers_only:
        raise NotImplementedError
    ins = inspect(engine)
    columns = ins.get_columns(table, schema)
    res = [{'name': c['name'], 'type': c['type'].python_type} for c in columns]
    for r in res:
        tokens, spaces = zip(*[(c.text.lower(), c.whitespace_) for c in nlp(r['name'])])
        r.update({'tokens': tokens, 'spaces': spaces})
    return res


def index_table(df, index, conn, dtypes=None, streaming=True, chunksize=1000):
    """Index a pandas DataFrame into ES as a mapping.

    Parameters
    ----------
        df : pandas.DataFrame
            The dataframe to be indexed
        index : str
            Name of the index in which to insert the table.
        conn : elasticsearch.client.Elasticsearch
            The ES client / connection to use.
        dtypes : mapping of NumPy datatypes to ES field types.
            Leaving it empty (default) is not recommended,
            since it will affect the quality of search results.
        streaming : bool
            Whether to use the streaming ES bulk API. Useful for large datasets.
        chunksize : int
            The chunksize to use when streaming,
            ignored if streaming is False or chunksize > len(df).
    Returns
    -------
        tuple
            A 2-tuple containing (number of records successfully indexed, number of failures).
            Ideally this should be (len(df), 0)

    Example
    -------
    >>> import pandas as pd
    >>> from elasticsearch_dsl import field
    >>> from elasticsearch_dsl.connections import create_connection
    >>> df = pd.read_csv('iris.csv')
    >>> dtypes = {'Petal Length': field.Double(),
    ...           'Sepal Width': field.Double(),
    ...           'Species': field.Keyword()}
    >>> conn = create_connection(hosts=['localhost'])
    >>> index_table(df, 'iris', conn, dtypes)
    (150, 0)

    """
    m = Mapping()
    if dtypes is None:
        warn(('Attempting to find ES types for the dataframe. This may affect search results.'
              ' Consider manually mapping the types to ES field types.'))
        dtypes = _find_es_types(df)
    for c, estype in dtypes.items():
        m.field(c, estype)
    m.save(index)

    def _actions():
        for i, r in df.iterrows():
            yield {'_index': index, '_source': r.to_dict()}

    if chunksize < df.shape[0] and streaming:
        status = [c for c in streaming_bulk(conn, _actions(), chunksize)]
        n_success = sum([r[0] for r in status])
        return n_success, df.shape[0] - n_success
    return bulk(conn, _actions(), stats_only=True)


def ents(doc, matcher):
    entities = list(doc.noun_chunks) + list(doc.ents)
    entities.extend([c for c in doc if c.pos_ in ('NOUN', 'PROPN')])
    for _, start, end in matcher(doc):
        entities.append(doc[start:end])
    return unoverlap(set(entities))


class TableSearch(object):
    """Prepare a table / ES index for search."""
    def __init__(self, nlp, index=None, mappings=None):
        """Create a search object specific to an ES index.

        Parameters
        ----------
        nlp : spacy.lang.en.English
            spaCy model.
        index : elasticsearch_dsl.Index, optional
            The elasticsearch_dsl Index object associated with the index to query.
        mappings: dict, optional
            The mapping of field names to ES types.
        One of `index` and `mappings` is required.

        Example
        -------
        >>> from elasticsearch_dsl.connections import create_connection
        >>> from elasticsearch_dsl import Index
        >>> from spacy import load
        >>> nlp = load('en_core_web_lg')
        >>> conn = create_connection(hosts=['localhost'])  # This step is important
        >>> index = Index('index name')
        >>> search_engine = TableSearch(nlp, index)
        >>> search_engine.query('Average price of coffee in Kerala')
        """
        self.nlp = nlp
        self.matcher = make_matcher(nlp)
        if index:
            self.mappings = index.get_mapping()[index._name]['mappings']['properties']
            self.index = index
        else:
            self.mappings = mappings
        if not self.mappings:
            raise ValueError('One of `index` and `mappings` is required.')
        self.coldocs = [self.nlp(c) for c in self.mappings]

    def search(self, indicators, entities, src_filters=None):
        body = {
            'query': {'bool': {'should': [
                {'multi_match': {'query': e.text}} for e in entities if e not in indicators
            ]}
            },
            'highlight': {
                'fields': {k: {} for k, v in self.mappings.items() if v in ('text', 'keyword')}}}
        s = Search(index=self.index._name)
        s = s.update_from_dict(body)
        return s.execute()

    def query(self, s, fallback=True):
        # TODO: Map fields to entity types supported by the model,
        # and restrict recognized entities to those fields.
        doc = s
        if not isinstance(s, spacy.tokens.Doc):
            doc = self.nlp(s)
        entities = ents(doc, self.matcher)
        indicators = self.find_classes(entities)
        if fallback:
            indicators.update(self.find_classes([c for c in doc if c.pos_ in ('NOUN', 'PROPN')]))
        indicators = {k: indicators[k] for k in unoverlap(indicators.keys())}
        return self.search(indicators, entities)

    def save(self, outpath):
        """Save the table search engine for later usage.

        Parameters
        ----------
        outpath : path to save to.

        Example
        -------
        >>> ts = TableSearch(nlp, index)
        >>> ts.save('/tmp/search-engine.zip')
        """
        outdir = tempfile.mkdtemp()
        try:
            self.nlp.to_disk(op.join(outdir, 'model'))
            with open(op.join(outdir, 'mapping.json'), 'w', encoding='utf8') as fout:
                json.dump(self.mappings, fout, indent=4)
            with zipfile.ZipFile(outpath, 'w') as zout:
                for root, dirs, files in os.walk(outdir):
                    for _file in files:
                        fpath = op.join(root, _file)
                        zout.write(fpath, arcname=op.relpath(fpath, outdir))
        finally:
            rmtree(outdir)

    @classmethod
    def load(cls, path):
        """Load a table search instance from disk.

        Parameters
        ----------
        path : str
            Path from which to load the instance.

        Example
        -------
        >>> ts = TableSearch.load('/tmp/search-engine.zip')
        """
        outdir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(outdir)
            nlp = spacy.load(op.join(outdir, 'model'))
            with open(op.join(outdir, 'mapping.json'), encoding='utf8') as fin:
                mappings = json.load(fin)
            return cls(nlp, mappings=mappings)
        finally:
            rmtree(outdir)

    def find_classes(self, ents, threshold=0.7):
        """Find which entity recognized in the input query matches a column name.

        An "named entity class" is any token that is synonymous with one of the column headers.

        Parameters
        ----------
        ents : sequence of spacy.token.span.Span objects
            Sequence of entities found in the input query.
        sim_threshold : float, optional
            Value at which to threshold cosine similarity scores
            in order to decide if a pair of tokens are synonymous.

        Returns
        -------
        Sequence of spacy.token.span.Span objects.
            Subset of the input sequence which contains only NE classes.
        """
        res = {}
        for e in ents:
            sims = sorted([(c, e.similarity(c)) for c in self.coldocs], key=lambda x: -x[1])
            sims = [(c, s) for c, s in sims if s > threshold]
            if sims:
                res[e] = sims[0][0]
        return res

    def find_instances(self, ents):
        """Find which entity recognized in the input query matches a column value.

        An "named entity instance" is any token that is a type or instance of a class name.
        E.g: "Commodity" is a column name, hence a named entity class, and "Coffee" is an
        instance of this class.

        Parameters
        ----------
        ents : sequence of spacy.token.span.Span objects
            Sequence of entities found in the input query.

        Returns
        -------
        Sequence of spacy.token.span.Span objects.
            Subset of the input sequence which contains only NE instances.
        """
        raise NotImplementedError

    def set_nes(self, field2nes):
        """Assign named entity labels to fields in the index.

        Parameters
        ----------
        field2nes : dict mapping {field_name: spacy entity label}
        """
        bad_labels = [c for _, c in field2nes.items() if c not in self.nlp.entity.labels]
        if bad_labels:
            raise ValueError('{} are not valid spaCy named entity labels.')
        self.field2nes = field2nes

    def _similarity(self, s):
        """Compute similarity of a string with each index fields.
        Only for debugging."""
        doc = self.nlp(s)
        sim = {c.text: doc.similarity(c) for c in self.coldocs}
        return pd.Series(sim).sort_values(ascending=False)
