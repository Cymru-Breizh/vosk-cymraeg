import argparse
import logging
import re
from pathlib import Path
from typing import List

import polars as pl
from rich.logging import RichHandler

import vosk_cymraeg.datasets.techiaith_text as techiaith_text
from vosk_cymraeg.normalisation import get_non_domain_chars, normalise_sentence
from vosk_cymraeg.phonetics.phonemizer import CyPhonemizer, EnPhonemizer, Phonemizer

URL_PATTERN = re.compile(
    r"([a-z]+:\/\/)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

_logger = logging.getLogger(__name__)


def main() -> None:
    """Create a training/dev/test corpus for Kaldi"""
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    args = _get_args()
    if len(args.lang) == 0:
        raise ValueError("You need to provide at least one language")

    output_folder = Path("data/output")

    # Load merged corpora
    _logger.info("Loading training set from disk")
    train_dataset = load_dataset(args.train, args.lang)

    # Load sentences from the training dataset (all should be Welsh)
    _logger.info("Loading sentences from training set and additional sources")
    sentences = pl.DataFrame(
        {"sentence": list(train_dataset["sentence"].unique()), "lang": "cy"}
    )

    # Since the sentences are already in the dataframe, we can remove the audio clips
    if args.enwau_text_only:
        train_dataset = train_dataset.filter(~pl.col("speaker").str.starts_with("enw"))

    # Load additional sentences from the tts prompts dataset and other sources
    if args.additional_text:
        sentences = pl.concat(
            [
                sentences,
                techiaith_text.load_techiaith_cofnodycynulliad_en_cy(),
                techiaith_text.load_techiaith_legislation_gov_uk_en_cy(),
                techiaith_text.load_techiaith_llyw_cymru_en_cy_ogl(),
                techiaith_text.load_str20tbl_tts_prompts_cy_en(),
                techiaith_text.load_wanasash_brawddegau_enwau_lleoedd(),
            ]
        )

    _logger.info(
        f"Loaded {len(sentences):,} sentences. Continuing on to normalising, filtering, and deduplicating the sentences"
    )
    sentences = (
        sentences.filter(pl.col("lang").is_in(args.lang))
        .unique()
        .filter(
            pl.col("sentence").map_elements(
                lambda x: URL_PATTERN.search(x) is None, pl.Boolean
            )
        )
        .with_columns(pl.col("sentence").map_elements(normalise_sentence, pl.String))
        .filter(pl.col("sentence").map_elements(filter_sentence, pl.Boolean))
    )
    _logger.info(f"The final text corpus contains {len(sentences):,} sentences")

    # Generate a word list based on the sentences in the sentences dataframe
    _logger.info("Generating word list based on sentences")
    words = (
        sentences.lazy()
        .with_columns(pl.col("sentence").str.split(" "))
        .explode("sentence")
        .unique()
        .sort(["lang", "sentence"])
        .rename({"sentence": "word"})
        .collect()
    )

    _logger.info(f"Loaded {len(words):,} number of words")

    # We only provide the train dataset to build the text corpus
    build_text_corpus(sentences["sentence"].unique(), output_folder)

    phones = build_lexicon(words, output_folder)

    # nonsilence_phones.txt
    nonsilence_phones_path = output_folder / "local/dict_nosp/nonsilence_phones.txt"
    with open(nonsilence_phones_path, "w", encoding="utf-8") as f:
        for p in sorted(phones):
            f.write(f"{p}\n")

    # silence_phones.txt
    silence_phones_path = output_folder / "local/dict_nosp/silence_phones.txt"
    with open(silence_phones_path, "w", encoding="utf-8") as f:
        f.write("SIL\noov\nSPN\nLAU\nNSN\n")  # TODO: remove 'oov' from silence phones ?

    # optional_silence.txt
    optional_silence_path = output_folder / "local/dict_nosp/optional_silence.txt"
    with open(optional_silence_path, "w", encoding="utf-8") as f:
        f.write("SIL\n")

    # Build files specific to train/dev/test datasets
    build_dataset("train", train_dataset, output_folder)

    # dev/test are optional: data prepared without a test split (when the
    # final evaluation runs outside the recipe) skips quietly when the CSV
    # doesn't exist.
    if args.dev and args.dev.exists():
        build_dataset(
            "dev",
            load_dataset(args.dev, args.lang),
            output_folder,
        )
    elif args.dev:
        _logger.warning(f"Dev dataset {args.dev} not found — skipping")

    if args.test and args.test.exists():
        build_dataset(
            "test",
            load_dataset(args.test, args.lang),
            output_folder,
        )
    elif args.test:
        _logger.warning(f"Test dataset {args.test} not found — skipping")


def filter_sentence(sentence: str) -> bool:
    if not sentence:
        return False
    invalid_chars = get_non_domain_chars(sentence)
    if invalid_chars:
        _logger.debug(f'[yellow]Invalid chars [{"".join(invalid_chars)}] "{sentence}"')
        return False
    return True


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "export",
        description="Script responsible for exporting Kaldi datasets",
    )

    # Used to wipe the output folder before processing
    parser.add_argument(
        "--train",
        default=Path("data/processed/dataset/train.csv"),
        help="Path to training dataset csv file",
        type=Path,
    )
    parser.add_argument(
        "--dev",
        default=Path("data/processed/dataset/dev.csv"),
        help="Path to development dataset csv file",
        type=Path,
    )
    parser.add_argument(
        "--test",
        default=Path("data/processed/dataset/test.csv"),
        help="Path to evaluation dataset csv file",
        type=Path,
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        help="Languages to export",
        default=["cy", "en"],
        choices=["cy", "en"],
    )
    parser.add_argument("--clear", action="store_true", help="Clears the target folder")
    parser.add_argument(
        "--additional_text", action="store_true", help="Add texts from other sources"
    )
    parser.add_argument(
        "--enwau_text_only",
        action="store_true",
        help="Remove enwau audio data from training set",
    )
    # parser.add_argument("--output", default="output", help="Target folder for the Kaldi dataset", type=Path)

    return parser.parse_args()


def load_dataset(path: Path, langs: list[str]) -> pl.DataFrame:
    return (
        pl.read_csv(path)
        .filter(pl.col("lang").is_in(langs))
        .with_columns(
            pl.col("sentence").map_elements(normalise_sentence, return_dtype=str)
        )
        .filter(pl.col("sentence").map_elements(filter_sentence, return_dtype=bool))
        .sort("utterance")
    )


def build_text_corpus(sentences: List[str], output_path: Path) -> None:
    """Build the text corpus"""

    output_path = output_path / "local"
    output_path.mkdir(exist_ok=True, parents=True)

    _logger.info("Building 'corpus.txt'")

    with open(output_path / "corpus.txt", "w", encoding="utf-8") as _f:
        for s in sorted(sentences):
            _f.write(f"{s}\n")


def build_lexicon(words: pl.DataFrame, output_path: Path) -> None:
    """
    Build a lexicon from a list of words

        Silences            -> SIL
        Spoken noises       -> SPN
        Non-spoken noises   -> NSN
        Laughter            -> LAU
    """

    special_tags = {
        "<anadlu>": "SPN",
        "<anadlu_i_mewn_yn_sydyn>": "SPN",
        "<aneglur>": "SPN",
        "<cerddoriaeth>": "NSN",
        "<chwerthin>": "LAU",
        "<chwibanu>": "SPN",
        "<chwythu_allan>": "SPN",
        "<clapio>": "NSN",
        "<clirio_gwddf>": "SPN",
        "<cusanu>": "SPN",
        "<distawrwydd>": "SIL",
        "<ochneidio>": "SPN",
        "<PII>": "SPN",
        "<peswch>": "SPN",
        "<sniffian>": "SPN",
        "<twtian>": "SPN",
    }

    _logger.info("Building 'lexicon.txt'")

    output_path = output_path / "local/dict_nosp"
    output_path.mkdir(exist_ok=True, parents=True)

    phone_set = set()
    phonemizers: dict[str, Phonemizer] = {"cy": CyPhonemizer(), "en": EnPhonemizer()}

    def get_pronunciation(data) -> list[list[str]]:
        word = data["word"]
        langs = data["lang"]
        pronunciations = [
            pronunciation
            for lang in langs
            for pronunciation in phonemizers[lang].phonemize(word)
        ]

        # Remove empty phones
        pronunciations = [[char for char in pron if char] for pron in pronunciations]
        # Remove empty pronunciations
        pronunciations = [pron for pron in pronunciations if pron]
        if len(pronunciations) == 0 and "cy" in langs:
            _logger.warning(f"Failed to phonemize {word!r}")

        return pronunciations

    # We need to group these by the word because we need to
    # produce all of the pronunciations for the same word
    # at the same time to make sure that there are no duplicates
    words = (
        words.lazy()
        .group_by("word")
        .all()
        .sort("word")
        .with_columns(
            pl.struct("word", "lang")
            .map_elements(get_pronunciation, pl.List(pl.List(pl.String)))
            .alias("pronunciation")
        )
        .filter(pl.col("pronunciation").list.len() > 0)
        .drop("lang")
        .explode("pronunciation")
        .unique(["word", "pronunciation"])
        .sort("word", pl.col("pronunciation").list.join(" "))
        .collect()
    )

    with open(output_path / "lexicon.txt", "w", encoding="utf-8") as _f:
        # Write special phones
        _f.write("!SIL SIL\n<UNK> SPN\n")
        for tag in sorted(special_tags.keys()):
            _f.write(f"{tag} {special_tags[tag]}\n")

        # Write regular words with corresponding phones
        for word, pronunciation in words.rows():
            _f.write(f"{word} {' '.join(pronunciation)}\n")
            phone_set.update(pronunciation)

    return phone_set


def build_dataset(name: str, df: pl.DataFrame, output_path: Path) -> None:
    """Generate Kaldi data for one sub-corpus, should be called for each split"""

    dataset_path = output_path / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Building '{name}' dataset")

    # Build 'text' file
    with open(dataset_path / "text", "w", encoding="utf-8") as _f:
        for row in df.rows(named=True):
            sentence = row["sentence"].strip()
            _f.write(f"{row['utterance']}\t{sentence}\n")

    # Build 'utt2spk'
    with open(dataset_path / "utt2spk", "w", encoding="utf-8") as _f:
        for row in df.sort("utterance").rows(named=True):
            _f.write(f"{row['utterance']}\t{row['speaker']}\n")

    # Build 'wav.scp'
    with open(dataset_path / "wav.scp", "w", encoding="utf-8") as _f:
        for row in df.sort("path").rows(named=True):
            audio_path = Path(row["path"]).resolve()
            _f.write(f"{row['utterance']}\t{audio_path.resolve()}\n")
