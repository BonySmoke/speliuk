# Speliuk

A more accurate spelling correction for the Ukrainian language.

## Motivation

When using a spell checker in systems that perform an automatic spelling correction without human verification, the following questions arise:
- How to avoid false correction, i.e. when a real word that is not present in a vocabulary is corrected? This is especially viable for fusional languages such as Ukrainian.
- How to find a single best correction for a misspelled word? Many spell checkers rely on the frequency of candidates and their edit distance discarding the surrounding context.

To address these issues, we propose a system that is compatible with any spell checker but focuses on precision over recall.<br>
We improve the accuracy of a spell checker by using these complimentary models:
- [KenLM](https://github.com/kpu/kenlm). The model is used for fast perplexity calculation to find the best candidate for a misspelled word.
- Transfomer-based NER pipeline to detect misspelled words.
- [SymSpell](https://github.com/wolfgarbe/SymSpell). As of now, this is the only supported spell checker.

## Installation

1. For CPU-only inference, install the CPU version of [PyTorch](https://pytorch.org/get-started/locally/).
2. Make sure you can compile Python extension modules (required for KenLM). If you are on Linux, you can install them like this:
```
sudo apt-get install python-dev
```
3. Install Speliuk:
```
pip install speliuk
```

## Usage

By default, Speliuk will use pre-trained models stored on [Hugging Face](https://huggingface.co/BonySmoke/Speliuk/tree/main).

```python
>>> from speliuk.correct import Speliuk
>>> speliuk = Speliuk()
>>> speliuk.load()
>>> speliuk.correct("то він моее це зраабити для меніе?")
Correction(corrected_text='то він може це зробити для мене?', annotations=[Annotation(start=7, end=11, source_text='моее', suggestions=['може'], meta={}), Annotation(start=15, end=23, source_text='зраабити', suggestions=['зробити'], meta={}), Annotation(start=28, end=33, source_text='меніе', suggestions=['мене'], meta={})])
```

Speliuk can also be used directly from a spaCy model:
```python
>>> import spacy
>>> from speliuk.correct import CorrectionPipe
>>> nlp = spacy.blank('uk')
>>> nlp.add_pipe('speliuk', config=dict(spacy_spelling_model_path='/my/custom/model'))
>>> doc = nlp("то він моее це зраабити для меніе?")
>>> doc._.speliuk_corrected
'то він може це зробити для мене?'
>>> doc.spans["speliuk_errors"]
[моее, зраабити, меніе]
```

## Training Details

### Spelling Error Detection

To detect spelling errors, a spaCy NER model is used.

It was trained on a combination of synthetic and golden data:
- For synthetic data generation, we used [UberText](https://lang.org.ua/en/ubertext/) as base texts and [nlpaug](https://github.com/makcedward/nlpaug) for errors generation. In total, 10k samples from different categories were used.
- For golden data, we used spelling errors from the [UA-GEC](https://github.com/grammarly/ua-gec) corpus.

### Perplexity Calculation

We used KenLM for quick perplexity calculation. We used an existing model [Yehor/kenlm-uk](https://huggingface.co/Yehor/kenlm-uk) trained on UberText.

### Spell Checker

We used [SymSpell](https://github.com/wolfgarbe/SymSpell) for error correction. The dictionary consists of 500k most frequent words from the UberText corpus.
