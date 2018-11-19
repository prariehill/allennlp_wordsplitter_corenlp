from typing import List

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter
from nltk.parse.corenlp import CoreNLPParser, CoreNLPServer
from overrides import overrides

__all__ = ['CorenlpSubprocWordSplitter', 'CorenlpRemoteWordSplitter']


@WordSplitter.register('corenlp_subproc')
class CorenlpSubprocWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses CoreNLP's tokenizer.
    It starts ``corenlp-server`` as a sub-process, and call it's Web API.
    """

    def __init__(self,
                 path_to_jar: str = None,
                 path_to_models_jar: str = None,
                 verbose: str = False,
                 java_options: str = None,
                 corenlp_options: str = None,
                 port: int = None,
                 encoding: str = 'utf8',
                 ):
        """
        Parameters
        ----------

        * For parameters from ``path_to_jar`` to ``port``, see https://www.nltk.org/api/nltk.parse.html#nltk.parse.corenlp.
        * For parameters ``encoding``,  see https://www.nltk.org/api/nltk.parse.html#nltk.parse.corenlp.CoreNLPParser
        """
        self._server = CoreNLPServer(
            path_to_jar, path_to_models_jar, verbose, java_options, corenlp_options, port)
        self._make_parser = lambda: CoreNLPParser(
            self._server.url, encoding, 'pos')
        self._server.start()

    def __del__(self):
        self._server.stop()

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        parser = self._make_parser()
        return [Token(t) for t in parser.tokenize(sentence)]


@WordSplitter.register('corenlp_remote')
class CorenlpRemoteWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses CoreNLP's tokenizer.
    It calls ``corenlp-server``'s Web API.
    """

    def __init__(self,
                 url: str = 'http://localhost:9000',
                 encoding: str = 'utf8',
                 ):
        """
        Parameters
        ----------

        see https://www.nltk.org/api/nltk.parse.html#nltk.parse.corenlp.CoreNLPParser
        """
        self._make_parser = lambda: CoreNLPParser(url, encoding, 'pos')

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        parser = self._make_parser()
        return [Token(t) for t in parser.tokenize(sentence)]
