import spacy
import random
import re
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from spacy.language import Language, Doc
from spacy.tokens import Token, DocBin
from spacy.cli.train import train
from nlpaug.augmenter.char import CharAugmenter
from ua_gec import Corpus, AnnotatedText, Document
from ua_gec.annotated_text import Annotation
from tqdm import tqdm

ALPHABET = "оаниітврсулдкмпяьзчбегхшжєйїцюфщґОАНИІТВРСУЛДКМПЯЬЗЧБЕГХШЖЄЙЇЦЮФЩҐ-'"


class CommonSpellingErrors:

    COMMON_ERROR_MAPPING = {
        "ьйо": "ьо",
        "жч": "щ",
        "дськ": "цьк",
        "щ": "жч"
    }

    COMMON_ERROR_REGEX = re.compile(r'.*(ьйо|жч|дськ|щ).*')

    def __init__(self, min_length: int = 3) -> None:
        self.min_length = min_length

    def common_errors(self, token: str):
        for correct, wrong in self.COMMON_ERROR_MAPPING.items():
            if correct in token:
                return token.replace(correct, wrong)

        return token

    def apostrophe(self, token: str):
        error = random.choice(["ь", ""])
        return token.replace("'", error)

    def merge_hyphen(self, token: str):
        return token.replace(" ", "-")

    def merge_split(self, token: str):
        return token.replace("-", " ")

    def excessive_soft_sign(self, token: str):
        return token.rstrip('ь')

    def uk_e_soft(self, token: str):
        return token.replace("е", "є")

    def uk_e_hard(self, token: str):
        return token.replace("є", "е")

    def uk_g_special(self, token: str):
        return token.replace("г", "ґ", 1)

    def uk_g(self, token: str):
        return token.replace("ґ", "г", 1)

    def split_ne(self, token: str):
        return token.replace("не", "не ")

    def merge_ne(self, token: str):
        return token.replace("не ", "не")

    def split_na(self, token: str):
        return token.replace("на", "на ")

    def merge_na(self, token: str):
        return token.replace("на ", "на")

    def _spelling_transformations_dispatch(self) -> dict[tuple]:
        return {
            "COMMON_ERRORS": (self.COMMON_ERROR_REGEX, self.common_errors),
            "EXCESSIVE_SOFT_SIGN": (re.compile(r'.*ь.*'), self.excessive_soft_sign),
            "є": (re.compile(r'.*е.*'), self.uk_e_soft),
            "е": (re.compile(r'.*є.*'), self.uk_e_hard),
            "ґ_1": (re.compile(r'.*г.*'), self.uk_g_special),
            "г_1": (re.compile(r'.*ґ.*'), self.uk_g),
            "SPLIT_NE": (re.compile(r'не.*'), self.split_ne),
            "SPLIT_NA": (re.compile(r'на.*'), self.split_na),
            "APOSTROPHE": (re.compile(r'.*ь.*'), self.apostrophe),
        }

    def augment(self, token: str):
        if len(token) < self.min_length:
            return token
        augmentations = self._spelling_transformations_dispatch()
        matching_transformations = list()
        for _, attributes in augmentations.items():
            regex: re.Pattern = attributes[0]
            func = attributes[1]
            if regex.match(token):
                matching_transformations.append(func)

        if not matching_transformations:
            return token

        transformation = random.choice(matching_transformations)
        error: str = transformation(token)
        # check if the token contains an error
        if token != error:
            return error

        return token


class SyntheticData:

    def __init__(self, data_filepath: str) -> None:
        self.data_filepath = data_filepath
        self.augmenters = self.load_augmenters()
        self.common_error_augmenter = CommonSpellingErrors()
        self.nlp = self.load_nlp()

    def load_nlp(self):
        if not Token.has_extension("error"):
            Token.set_extension("error", default="")

        nlp = spacy.load("uk_core_news_sm", enable=["tokenizer"])
        nlp.add_pipe("errorifier")

        return nlp

    def load_augmenters(self):
        augmenters: list[CharAugmenter] = list()
        keyboard_aug = nac.KeyboardAug(
            lang="uk", aug_char_max=2, aug_char_p=0.3)
        augmenters.append(keyboard_aug)

        random_char_actions = ['swap', 'delete']
        for action in random_char_actions:
            char_augmenter = nac.RandomCharAug(
                action=action, aug_char_max=2, aug_char_p=0.3)
            augmenters.append(char_augmenter)

        insert_augmenter = nac.RandomCharAug(
            action="insert", aug_char_max=2, aug_char_p=0.3,
            candidates=[c for c in ALPHABET])

        split_augmenter = naw.SplitAug()
        augmenters.extend([insert_augmenter, split_augmenter])

        return augmenters

    @Language.component("errorifier")
    def errorifier(self, doc: Doc):
        """Randomly generate errors for tokens"""
        # whether to errorify a token
        options = [True, False]
        probs = [0.3, 0.7]
        for token in doc:
            # emoji
            if str(token).startswith(':') and str(token).endswith(':'):
                token._.error = str(token)
                continue

            option = random.choices(options, probs)[0]
            if not option:
                token._.error = str(token)
                continue

            should_run_common = random.choices(options, probs)[0]
            if should_run_common:
                common_error = self.common_error_augmenter.augment(str(token))
                if common_error != str(token):
                    token._.error = common_error
                    continue

            random_augmenter: CharAugmenter = random.choice(self.augmenters)
            augmented_text = random_augmenter.augment(str(token))
            token._.error = augmented_text[0]

        return doc

    def get_spacy_docs_with_error_tokens(self):
        with open(self.data_filepath, 'r') as file:
            texts = [line.strip() for line in file if line.strip()]

        docs = list(tqdm(self.nlp.pipe(texts)))
        return docs

    def get_error_annotated_documents(self):
        docs = self.get_spacy_docs_with_error_tokens()
        error_docs = list()
        for doc in docs:
            words = [token._.error for token in doc]
            spaces = [bool(token.whitespace_) for token in doc]
            error_doc = Doc(self.nlp.vocab, words=words, spaces=spaces)

            error_spans = list()
            for i in range(len(error_doc)):
                original_token: Token = doc[i]
                error_token: Token = error_doc[i]

                # there is no error
                if original_token.text == error_token.text:
                    continue

                start, end = error_token.idx, error_token.idx + \
                    len(error_token.text)
                error_span = error_doc.char_span(start, end, label="SPELLING")
                if error_span.text.startswith(' ') or error_span.text.endswith(' '):
                    continue
                error_spans.append(error_span)

            # error_doc.ents = error_spans
            error_doc.spans["speliuk"] = error_spans
            error_docs.append(error_doc)

        return error_docs


class UaGecSpelling:
    """Get spelling errors from the ua-gec corpus"""

    def __init__(self) -> None:
        self.nlp = spacy.load("uk_core_news_sm", enable=["tokenizer"])

    def valid_annotation(self, annotation: Annotation):
        """Check if a given annotation is valid for spell checking"""
        skip_tokens = ["у", "в", "з", "і", "із", "й", "о", "об", "та"]
        if annotation.meta["error_type"] != "Spelling":
            return False
        if not annotation.top_suggestion:
            return False
        if not annotation.source_text:
            return False
        if annotation.source_text.isspace():
            return False
        if annotation.source_text.lower() in skip_tokens:
            return False
        if annotation.top_suggestion.lower() in skip_tokens:
            return False
        if annotation.source_text.lower().replace("у", "в", 1) == annotation.top_suggestion:
            return False
        if annotation.source_text.lower().replace("в", "у", 1) == annotation.top_suggestion:
            return False
        if annotation.source_text.capitalize() == annotation.top_suggestion:
            return False
        if annotation.source_text.lower() == annotation.top_suggestion.lower():
            return False

        return True

    def get_docs_with_spelling_annotations(self, corpus: Corpus):
        spelling_docs: list[Document] = []

        for doc in tqdm(corpus):
            text = doc.annotated

            for annotation in text.iter_annotations():
                if not self.valid_annotation(annotation):
                    text.remove(annotation)

            remaining_annotations = text.get_annotations()
            if not remaining_annotations:
                continue

            spelling_doc = Document(
                annotated=text,
                meta=doc.meta,
                partition_dir=doc._partition_dir
            )

            spelling_docs.append(spelling_doc)

        return spelling_docs

    def get_spelling_spacy_docs(self, corpus: Corpus):
        spelling_docs = self.get_docs_with_spelling_annotations(corpus)
        spelling_annotated_texts: list[AnnotatedText] = list()
        for doc in spelling_docs:
            spelling_texts = [
                AnnotatedText(text) for text in
                doc.annotated.get_annotated_text().split('\n') if str(text)
            ]
            spelling_annotated_texts.extend(spelling_texts)

        spacy_docs = list()
        for text in tqdm(spelling_annotated_texts):
            doc = self.nlp(text.get_original_text())
            error_spans = list()
            for annotation in text.iter_annotations():
                annotation: Annotation = annotation
                start, end = annotation.start, annotation.end
                if annotation.source_text.startswith(' '):
                    start = end - len(annotation.source_text.lstrip())
                if annotation.source_text.endswith(' '):
                    end = start + len(annotation.source_text.rstrip())
                error_span = doc.char_span(start, end, label="SPELLING")
                if not error_span:
                    continue

                error_spans.append(error_span)

            # doc.ents = error_spans
            doc.spans["speliuk"] = error_spans
            spacy_docs.append(doc)

        return spacy_docs


class Trainer:

    def __init__(self, data_filepath: str, output_path: str) -> None:
        self.data_filepath = data_filepath
        self.output_path = output_path
        self.training_data = None
        self.test_data = None

    def load_data(self):
        self.training_data = self.get_training_data()
        self.test_data = self.get_test_data()

    def get_training_data(self):
        ua_gec_train_corpus = UaGecSpelling().get_spelling_spacy_docs(
            Corpus(partition='train'))
        synthetic_train_docs = SyntheticData(self.data_filepath)

        train_docs = synthetic_train_docs + ua_gec_train_corpus

        return train_docs

    def get_test_data(self):
        ua_gec_test_corpus = UaGecSpelling().get_spelling_spacy_docs(
            Corpus(partition='test'))

        return ua_gec_test_corpus

    def save_train_data(self):
        train_db = DocBin()
        [train_db.add(doc) for doc in self.training_data]
        train_db.to_disk(f"./{self.output_path}/spelling-train.spacy")

    def save_test_data(self):
        dev_db = DocBin()
        [dev_db.add(doc) for doc in self.test_data]
        dev_db.to_disk(f"./{self.output_path}/spelling-dev.spacy")

    def train_ner(self):

        train("./config.cfg", output_path=self.output_path, overrides={
            "paths.train": "./spelling-train.spacy", "paths.dev": "./spelling-dev.spacy"
        })
