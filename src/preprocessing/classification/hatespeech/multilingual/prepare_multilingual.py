import argparse
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandarallel import pandarallel
from os import path, listdir
from dynaconf import settings
from sklearn.model_selection import train_test_split

from src.inference.lang_dtc import lang_dtc_setup_fn
from src.preprocessing import get_clean_fn

np.random.seed(20)
pandarallel.initialize()


COLUMNS = [
    "comment_text",
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit"
]

SPECIAL_CHARACTERS = {
    "’": "'", "‘": "'", "´": "'", "`": "'", "…": "...", "&": " and ", "“": '"', "”": '"',
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i",
    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r",
    "s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z"
}
special_characters_re = re.compile('({})'.format('|'.join(SPECIAL_CHARACTERS.keys())))
special_characters_map = lambda match: SPECIAL_CHARACTERS[match.group(0)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", default="hatespeech")
    parser.add_argument("--db_version", default="0.1")
    parser.add_argument("--sentence_min_len", default=5)
    return parser.parse_args()

def _fix_labels_(x):
    if x > 0.5:
        return 1
    elif x <= 0.30:
        return 0
    else:
        return None


def preprocess_toxic_comment(args):
    def train_fn(
            db_path: str,
            db_file: str
    ):
        df = pd.read_csv(
            path.join(db_path, db_file),
            index_col="id"
        )

        df.rename(
            columns=dict(
                toxic="toxicity",
                severe_toxic="severe_toxicity",
                identity_hate="identity_attack"
            ),
            inplace=True
        )

        df.describe()
        print(df.head(10))

        # remove special characters
        df["comment_text"] = df["comment_text"].apply(
            lambda x: special_characters_re.sub(special_characters_map, x)
        )
        # remove short sentences
        print(f"shape before remove short sentences {df.shape}")
        df = df[df["comment_text"].apply(len) > args.sentence_min_len]
        print(f"shape after remove short sentences {df.shape}")

        for col in COLUMNS:
            if col in df.columns:
                print(f"{col}\t{df[col].isna().sum()}")

        # remove nan
        print(f"shape before remove nan {df.shape}")
        df.dropna(axis=0, inplace=True)
        print(f"shape after remove nan {df.shape}")

        # add missing column
        if "sexual_explicit" not in df.columns:
            df["sexual_explicit"] = np.ones_like(df["toxicity"]) * -1.

        # add language
        if "lang" not in df.columns:
            df["lang"] = "en"

        df.to_csv(
            path.join(
                db_path,
                db_file.replace(".csv", ".tsv")
            ),
            sep="\t"
        )

    def test_fn(
            db_path: str,
            db_file: str,
            db_labels: str
    ):
        df = pd.read_csv(
            path.join(
                db_path,
                db_file
            ),
            index_col="id"
        ).join(
            pd.read_csv(
                path.join(db_path, db_labels),
                index_col="id"
            )
        )

        df.rename(
            columns=dict(
                toxic="toxicity",
                severe_toxic="severe_toxicity",
                identity_hate="identity_attack"
            ),
            inplace=True
        )

        # remove special characters
        df["comment_text"] = df["comment_text"].apply(
            lambda x: special_characters_re.sub(special_characters_map, x)
        )
        # remove short sentences
        df = df[df["comment_text"].apply(len) > args.sentence_min_len]

        # remove row with -1
        df[(df["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"] == -1)
               .sum(axis=1) > 0] = None

        print(f"shape before remove nan {df.shape}")
        df.dropna(axis=0, inplace=True)
        print(f"shape after remove nan {df.shape}")

        # add missing column
        if "sexual_explicit" not in df.columns:
            df["sexual_explicit"] = np.ones_like(df["toxicity"]) * -1.

        # add language
        if "lang" not in df.columns:
            df["lang"] = "en"

        df.to_csv(
            path.join(
                db_path,
                db_file.replace(".csv", ".tsv")
            ),
            sep="\t"
        )

    return train_fn, test_fn


def preprocess_unintended_comment(args):
    lang_dtc_preprocess, lang_dtc_inference, lang_dtc_postprocess = lang_dtc_setup_fn(
        path.join(
            settings.get("ckp_dir"),
            "import",
            "lang-dtc",
            "lid.176.bin"
        )
    )

    def train_fn(
            db_path: str,
            db_file: str
    ):
        df = pd.read_csv(
            path.join(db_path, db_file),
        )

        df.set_index(
            "id",
            inplace=True
        )

        df.rename(
            columns=dict(
                toxic="toxicity",
            ),
            inplace=True
        )

        # only select needed columns
        df = df[COLUMNS]

        df.describe()
        print(df.head(10))

        # remove special characters
        df["comment_text"] = df["comment_text"].apply(
            lambda x: special_characters_re.sub(special_characters_map, x)
        )
        # remove short sentences
        print(f"shape before remove short sentences {df.shape}")
        df = df[df["comment_text"].apply(len) > args.sentence_min_len]
        print(f"shape after remove short sentences {df.shape}")


        for col in COLUMNS:
            if col in df.columns:
                is_nan = df[col].isna()
                print(f"{col}\t{is_nan.sum()}")

        # discretize labels
        for col in COLUMNS[1:]:
            df[col] = df[col].apply(_fix_labels_)

        # remove nan
        print(f"shape before remove nan {df.shape}")
        df.dropna(axis=0, inplace=True)
        print(f"shape after remove nan {df.shape}")



        # add language
        if "lang" not in df.columns:
            df["lang"] = df["comment_text"].parallel_apply(
                lambda x:
                lang_dtc_postprocess(
                    *lang_dtc_inference(
                        lang_dtc_preprocess(
                            [x]
                        )
                    )
                )[0][0]
            )
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df.groupby(["lang"]).count())

            # remove language with less than 5 examples
            collector = []
            for lang, g_index in df.groupby("lang"):
                if g_index.shape[0] > 15:
                    collector.append(g_index)

            df = pd.concat(collector, axis=0)
            print(f"shape after remove language {df.shape}")

        df.to_csv(
            path.join(
                db_path,
                db_file.replace(".csv", ".tsv")
            ),
            sep="\t"
        )

    def test_fn(
            db_path: str
    ):
        df = pd.concat(
            (
                pd.read_csv(
                    path.join(db_path, "test_private_expanded.csv"),
                    index_col="id"
                ),
                pd.read_csv(
                    path.join(db_path, "test_public_expanded.csv"),
                    index_col="id"
                )
            ),
            sort=True,
            axis=0
        )

        df.rename(
            columns=dict(
                toxic="toxicity",
            ),
            inplace=True
        )

        df = df[COLUMNS]

        # remove special characters
        df["comment_text"] = df["comment_text"].apply(
            lambda x: special_characters_re.sub(special_characters_map, x)
        )
        # remove short sentences
        df = df[df["comment_text"].apply(len) > args.sentence_min_len]

        # discretize labels
        for col in COLUMNS[1:]:
            df[col] = df.apply(_fix_labels_)

        for col in COLUMNS:
            if col in df.columns:
                print(f"{col}\t{df[col].isna().sum()}")

        # remove nan
        print(f"shape before remove nan {df.shape}")
        df.dropna(axis=0, inplace=True)
        print(f"shape after remove nan {df.shape}")

        # add language
        if "lang" not in df.columns:
            df["lang"] = df["comment_text"].apply(
                lambda x:
                lang_dtc_postprocess(
                    lang_dtc_inference(
                        lang_dtc_preprocess(
                            x
                        )
                    )
                )[0]
            )

        df.to_csv(
            path.join(
                db_path,
                "test.tsv"
            ),
            sep="\t"
        )

    return train_fn, test_fn


if __name__ == '__main__':
    args = parse_args()

    # toxic_train_fn, toxic_test_fn = preprocess_toxic_comment(args)
    # toxic_train_fn(
    #     path.join(
    #         settings.get("data_dir"),
    #         args.db_name
    #     ),
    #     "jigsaw-toxic-comment-train.csv"
    # )
    #
    # unintended_train_fn, unintended_test_fn = preprocess_unintended_comment(args)
    # unintended_train_fn(
    #     path.join(
    #         settings.get("data_dir"),
    #         args.db_name
    #     ),
    #     "jigsaw-unintended-bias-train.csv"
    # )

    df = pd.concat((
        pd.read_csv(
            path.join(
                settings.get("data_dir"),
                args.db_name,
                "jigsaw-unintended-bias-train.tsv"
            ),
            sep="\t",
        ), pd.read_csv(
            path.join(
                settings.get("data_dir"),
                args.db_name,
                "jigsaw-toxic-comment-train.tsv"
            ),
            sep="\t"
        )),
        axis=0
    )

    # get cleaning function
    clean_fn = get_clean_fn(
        lambda text: text
    )
    # clean data
    df["comment_text"] = df.parallel_apply(
        lambda x: clean_fn(
            data=x["comment_text"],
            lang=x["lang"],
            lowercase=False
        ),
        axis=1
    )
    # remove duplicate examples
    df.drop_duplicates(inplace=True)

    # print stats
    for col in COLUMNS + ["lang"]:
        assert df[col].isna().any() == False
        print(f"{col}\t distinct \t {df[col].unique().shape[0]} \t {df[col].shape[0]} \t {df[col].unique().shape[0] / df[col].shape[0]}")

        if is_numeric_dtype(df[col]):
            print(f"{col}\t pos {(df[col] == 1).sum()}")
            print(f"{col}\t neg {(df[col] == 0).sum()}")
            print(f"{col}\t unk {(df[col] == -1).sum()}")


    # get train, test and eval set
    X_train, X_test, y_train, y_test = train_test_split(df[["comment_text", "lang"]], df[COLUMNS[1:]], shuffle=True, test_size=0.1)

    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, shuffle=True, test_size=0.1)

    assert (y_test[["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]].sum(axis=0) > 0).all() \
           and ((y_test["sexual_explicit"] >= 0).sum(axis=0) > 0).all()
    assert (y_eval[["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]].sum(axis=0) > 0).all() \
           and ((y_eval["sexual_explicit"] >= 0).sum(axis=0) > 0).all()
    assert (y_train[["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]].sum(axis=0) > 0).all() \
           and ((y_train["sexual_explicit"] >= 0).sum(axis=0) > 0).all()

    pd.concat((
        X_train, y_train
    ),
        axis=1
    ).to_csv(
        path.join(
            settings.get("data_dir"),
            args.db_name,
            "train.tsv"
        ),
        sep="\t"
    )

    pd.concat((
        X_eval, y_eval
    ),
        axis=1
    ).to_csv(
        path.join(
            settings.get("data_dir"),
            args.db_name,
            "eval.tsv"
        ),
        sep="\t"
    )

    pd.concat((
        X_test, y_test
    ),
        axis=1
    ).to_csv(
        path.join(
            settings.get("data_dir"),
            args.db_name,
            "test.tsv"
        ),
        sep="\t"
    )