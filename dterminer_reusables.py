"""Various functions that may be reusable throughout the D-Terminer project for different pipelines"""

import os
import cchardet as chardet


def listdir_nohidden(path, extension=""):
    """
    provide a list of all non-hidden filenames in a given directory path;
    Careful: returns file names, not filepaths!

    :param path: path to directory containing non-hidden files
    :param extension: if specified, check specifically for files that end with given extension
    :return: list of all filenames found in the given directory
    """
    for f in os.listdir(path):
        if not f.startswith('.') and not extension:
            yield f
        elif not f.startswith('.') and f.endswith(extension):
            yield f


def check_language(language, de=True):
    """
    Check whether the given language is appropriate, i.e., whether it is "nl", "fr", "en", or "de".
    If it is one of these, but uppercase, lowercase it. If it is the full form of one of these, e.g., "Dutch",
    change it into the appropriate abbreviation. Return the correct (lowercased, abbreviated) form of
    the language and raise an error if the language argument is not one supported by the script.

    :param language: "nl", "fr", "en", or "de"
    :param de: whether German is an accepted language
               (separate option since de is supported for preprocessing, but not part of training data)
    :return: the language lowercased and with the proper abbreviation, i.e.: "nl", "fr", "en", or "de"
    """
    l_dict = {"dutch": "nl", "french": "fr", "english": "en", "german": "de"}
    if de:
        if language.lower() not in ["nl", "fr", "en", "de"]:
            if language.lower() not in l_dict:
                raise ValueError(f"ERROR check_language: language parameter {language} not supported;\n"
                                 f"expected 'nl', 'fr', 'en', or 'de'!")
            else:
                l = l_dict[language]
        else:
            l = language.lower()
    else:
        if language.lower() not in ["nl", "fr", "en"]:
            if language.lower() not in ["english", "french", "dutch"]:
                raise ValueError(f"ERROR check_language: language parameter {language} not supported;\n"
                                 f"expected 'nl', 'fr', or 'en'!")
            else:
                l = l_dict[language]
        else:
            l = language.lower()
    return l


def check_corpus_dp(dp, extension):
    """
    Check whether the path to the directory that was provided is valid to start preparing data.
    The directory is expected to be subdirectory of "D-Terminer/unseen_corpora", with a
    main directory where all info on the corpus will be stored "D-Terminer/unseen_corpora/[corpus_name]/"
    (the given path (dp) should point to this main directory), and a subdirectory
    "D-Terminer/unseen_corpora/[corpus_name]/corpus/" where the text file(s) of the corpus are stored.
    During the term extraction process, different subdirectories will be created to store the preprocessed
    version of the corpus and the results. If dp is not a subdirectory of "unseen_corpora", a warning
    will be raised. If no corpus is stored under "corpus", an error will be raised.

    :param dp: path to directory where all info on corpus is stored
    :param extension: how the filenames in the dp should end (e.g., ".txt" or ".tmx")
    :return: dp (make sure it ends with "/") or error if necessary
    """
    if not dp.endswith("/"):
        dp += "/"
    # check whether directory exists
    if not os.path.exists(dp):
        raise ValueError(f"ERROR check_corpus_dp: given path to directory (dp) leads to directory that "
                         f"does not exist:\n{dp}")

    # check whether directory under "unseen_corpora"
    if not dp.startswith("unseen_corpora"):
        raise Warning(f"WARNING check_corpus_dp: given path to directory (dp) is not a subdirectory of "
                      f"'unseen_corpora' as expected:\n{dp}")

    # check whether there is a subdirectory called "corpus" in which text file(s) are stored
    corpus_dp = dp + "corpus/"
    if not os.path.exists(corpus_dp):
        raise ValueError(f"ERROR check_corpus_dp: given path to directory (dp) leads to directory that "
                         f"does not have a subdirectory called 'corpus',\n"
                         f"where the texts of the corpus should be stored:\n{dp}")
    else:
        corpus_filenames = listdir_nohidden(corpus_dp, extension=extension)
        if not corpus_filenames:
            raise ValueError(f"ERROR check_corpus_dp: given path to directory (dp) has subdirectory 'corpus'\n"
                             f"but no files are stored there: {corpus_dp}")
        else:
            for corpus_filename in corpus_filenames:
                if not corpus_filename.endswith(extension):
                    raise ValueError(f"ERROR check_corpus_dp: given path to directory (dp) "
                                     f"has subdirectory 'corpus',\nbut the files stored there "
                                     f"are not '.txt' files as expected: {corpus_dp}")
    return dp


def check_existing_data_seq_no_features(dp):
    """
    Check whether the corpus in the given main directory (probably subdir of "unseen_corpora")
    has not already been prepared. If a directory under the main dp already exists with
    the name "data_seq_no_features" and is not empty, ask if existing files should be removed.
    If directory does not exist yet, create it.

    :param dp: path to directory where all info on corpus is stored
    :return: "continue" or "stop" if data is prepared and should not be overwritten
    """
    data_dp = dp + "data_seq_no_features/"
    if not os.path.exists(data_dp):
        os.mkdir(data_dp)
        return "continue"
    elif len(list(listdir_nohidden(data_dp))) > 0:
        proceed = ""
        while proceed not in ["stop", "overwrite"]:
            proceed = input(f"INPUT REQUESTED check_existing_data_seq_no_features:\n"
                            f"the directory {data_dp}\n"
                            f"already exists and appears to contain files.\n"
                            f"Do you want to remove these files and prepare the corpus from scratch again?\n"
                            f"Or do you want to stop preprocessing/preparing the corpus?\n"
                            f"Type 'overwrite' or 'stop' to choose.")
        if proceed == "overwrite":
            for fn in listdir_nohidden(data_dp):
                fp = data_dp + fn
                os.remove(fp)
            return "continue"
        else:
            return "stop"


def get_sublist_indices(sublist, main_list):
    """
    Get the indices of a sublist in a larger list.

    :param sublist: smaller list
    :param main_list: larger list that contains sublist
    :return: nested list of all overlapping indices, e.g.: [[2, 3]]
    """
    results = []
    sublist_length = len(sublist)
    for ind in (i for i, e in enumerate(main_list) if e == sublist[0]):
        if main_list[ind:ind+sublist_length] == sublist:
            overlapping_indices = []
            for i in range(ind, ind+sublist_length):
                overlapping_indices.append(i)
            results.append(overlapping_indices)
    return results


def check_encoding(fp):
    """
    Use cchardet to check encoding of file end return the correct encoding

    :param fp: path to file
    :return: encoding (e.g., "UTF-8" or "UTF-16")
    """
    with open(fp, "rb") as f:
        content = f.read()
        result = chardet.detect(content)
        encoding = result["encoding"]
    return encoding

