"""
Multilingual term extraction pipeline for parallel corpora (.tmx),
based on the seq_bert_multi.py monolingual term extraction.

Author: Ayla Rigouts Terryn
Created: 28/03/2022
Last updated: 04/05/2022
"""

import os
import statistics
from lxml import etree
from astred import Aligner
from operator import itemgetter
from lxml.etree import ElementTree
from seq_bert_multi import prep_corpus_sbm, extract_terms_sbm
from dterminer_reusables import check_language, check_corpus_dp, check_existing_data_seq_no_features, \
    listdir_nohidden, get_sublist_indices, check_encoding


def remove_newlines_before_seg(tmx_fp, encoding):
    """
    Remove all newline characters before <seg> elements for easier processing
    rewrite file if necessary to same path with "_edited" appended

    :param tmx_fp: path to .tmx file
    :param encoding: encoding of file as found by check_encoding
    :return: correct fp: original one if nothing changed, edited one if it did
    """
    # Read in the file
    with open(tmx_fp, "rt", encoding=encoding) as tmx_f:
        content = tmx_f.read()

    # Replace the target string
    content_without_newline = content.replace(">\n<seg>", "><seg>")
    if content != content_without_newline:
        # Write the file out again
        tmx_fp_edited = tmx_fp + "_edited"
        with open(tmx_fp_edited, "wt", encoding=encoding) as tmx_f:
            tmx_f.write(content_without_newline)
            print(f"\t\t> removed unnecessary newlines and wrote to {tmx_fp_edited};\n"
                  f"process will be continued based on this file")
            return tmx_fp_edited
    else:
        return tmx_fp


def check_tmx_language_codes(tmx_fp, encoding):
    """
    Based on the language codes found in the tmx file,
    create a dictionary with the general language codes as keys ("en", "fr", "nl, "de")
    and the matching language codes in the tmx as values.
    return that dictionary.
    Also check the naming of the tuv segments as either "tuv lang" or "tuv xml:lang"
    and return tuv_name as either one.

    :param tmx_fp: path to .tmx file
    :param encoding: encoding of file as found by check_encoding
    :return: language_code_dict = {"en": "equivalent code in tmx", "fr": "...", "nl": "...", "de": "..."},
             tuv_name ('tuv lang' or 'tuv xml:lang')
    """
    language_code_dict = {"en": "", "fr": "", "nl": "", "de": ""}
    ignored_languages = []
    tuv_name = ""
    with open(tmx_fp, "rt", encoding=encoding) as tmx_f:
        lines = tmx_f.read().splitlines()
        for line in lines:
            xml_lang_index = line.find("lang=")
            if xml_lang_index > -1 and "tuv" in line:
                # check tuv_name
                if not tuv_name:
                    if "tuv lang" in line:
                        tuv_name = "tuv lang"
                    elif "tuv xml:lang" in line:
                        tuv_name = "tuv xml:lang"
                    else:
                        raise ValueError(
                            f"tuv name not 'tuv lang' or 'tuv xml:lang': how are segments with language codes named?")
                language_code_start_index = xml_lang_index + 6
                closing_bracket_index = line.find(">")
                if not closing_bracket_index > language_code_start_index:
                    print("ERROR check_tmx_language_codes: checkpoint 1 end index of language code field does not work")
                else:
                    language_code_end_index = closing_bracket_index - 1
                language_code = line[language_code_start_index:language_code_end_index]
                check = False
                for language_code_general, language_code_tmx in language_code_dict.items():
                    if language_code.lower().startswith(language_code_general.lower()):
                        check = True
                        if not language_code_tmx:
                            language_code_dict[language_code_general] = language_code
                        elif language_code_tmx != language_code:
                            print(f"ERROR check_tmx_language_codes: checkpoint 2 different language codes for same"
                                  f"language?\n"
                                  f"language code found: {language_code}\n"
                                  f"language code tmx dict: {language_code_tmx}\n"
                                  f"{language_code_dict}")
                if not check:
                    full_languages = {"dutch": "nl", "french": "fr", "english": "en", "german": "de"}
                    for full_language, language_code_general in full_languages.items():
                        if full_language in language_code.lower():
                            check = True
                            if not language_code_general:
                                language_code_dict[language_code_general] = language_code
                            elif language_code_tmx != language_code:
                                print(f"ERROR check_tmx_language_codes: checkpoint 3 different language codes for same"
                                      f"language? {language_code}\n{language_code_dict}")
                if not check:
                    if language_code not in ignored_languages:
                        ignored_languages.append(language_code)
    print(f"\t\t> detected language codes: {language_code_dict}\n")
    print(f"\t\t> ignoring language codes: {ignored_languages}\n")
    print(f"\t\t> tuv name: {tuv_name}\n")
    return language_code_dict, tuv_name


def tmx2txt(tmx_fp, language_dp_dict, verbose=False):
    """
    Based on a .tmx file, extract .txt files for each of the given languages
    and store the resulting .txt files (same filenames as original, but with _[language].txt)
    in the specified output directory.
    Note: the resulting .txt files will have the same alignments, so the same number of lines
    and each line can be aligned to the same line in the different language version.

    :param tmx_fp: path to .tmx file
    :param language_dp_dict: dictionary with as keys languages (language codes) to extract from the .tmx
                             (currently supported: ["en", "fr", "nl", "de"]); and as values the
                             paths to the directories where resulting .txt files should be saved
    :param verbose: whether to print intermediate info
    :return: nothing, but print progress
    """
    # check/create/get appropriate paths
    languages_fps_dict = {}
    fps_texts_dict = {}
    for l, out_dp in language_dp_dict.items():
        if l.lower() not in ["en", "fr", "nl", "de"]:
            raise ValueError(f"\nERROR tmx2txt: currently only supports 'en', 'fr', 'nl, and 'de'; not {l}\n")
        if not out_dp.endswith("/"):
            out_dp += "/"
        if not os.path.exists(out_dp):
            if verbose:
                print(f"* tmx2tx: given output directory does not exist yet, creating it now: {out_dp}\n")
            os.mkdir(out_dp)
        tmx_fn = tmx_fp.split("/")[-1]
        out_fp = out_dp + tmx_fn
        out_fp = out_fp.replace(".tmx", f"_{l.lower()}.txt")
        languages_fps_dict[l.lower()] = out_fp
        fps_texts_dict[out_fp] = ""
        if os.path.exists(out_fp):
            proceed = ""
            while proceed not in ["stop", "overwrite", "continue"]:
                proceed = input(f"INPUT REQUESTED tmx2txt:\n"
                                f"the file {out_fp} \n"
                                f"already exists and appears to contain files.\n"
                                f"Do you want to remove these files and get output from scratch again?"
                                f" > type 'overwrite'\n"
                                f"Or do you want to stop the term extraction process? > type 'stop'\n"
                                f"Or do you want to use the existing data? > type 'continue'"
                                f"Type 'overwrite', 'stop', or 'continue' to choose.")
            if proceed == "stop":
                raise ValueError(f"STOPPED tmx2txt because of pre-existing files")
            elif proceed == "overwrite":
                continue
            elif proceed == "continue":
                return

    # check encoding
    encoding = check_encoding(tmx_fp)

    # remove newlines if necessary
    tmx_fp = remove_newlines_before_seg(tmx_fp, encoding)

    # check language codes
    language_code_dict, tuv_name = check_tmx_language_codes(tmx_fp, encoding)

    # parse tmx
    if tuv_name == "tuv xml:lang":
        nsmap = {"xml": "http://www.w3.org/XML/1998/namespace"}
    else:
        nsmap = {"tmx": "http://www.lisa.org/tmx14"}
    try:
        tree: ElementTree = etree.parse(tmx_fp)
        tus = tree.findall("//tu")
        for tu_id, tu in enumerate(tus, 1):
            for language in language_dp_dict.keys():
                segment = ""
                if tuv_name == "tuv xml:lang" and \
                        tu.find(f"./tuv[@xml:lang='{language_code_dict[language]}']/seg", namespaces=nsmap) is not None:
                    segment = tu.find(f"./tuv[@xml:lang='{language_code_dict[language]}']/seg", namespaces=nsmap).text
                elif tuv_name == "tuv lang" and \
                        tu.find(f"./tuv[@lang='{language_code_dict[language]}']/seg", namespaces=nsmap) is not None:
                    segment = tu.find(f"./tuv[@lang='{language_code_dict[language]}']/seg", namespaces=nsmap).text
                if segment:
                    fps_texts_dict[languages_fps_dict[language.lower()]] += segment + "\n"
                else:
                    segment_exists_in_other_language = False
                    for l_inner in language_dp_dict.keys():
                        if tuv_name == "tuv xml:lang" and \
                                tu.find(f"./tuv[@xml:lang='{language_code_dict[l_inner]}']/seg", namespaces=nsmap) \
                                is not None:
                            if tu.find(f"./tuv[@xml:lang='{language_code_dict[l_inner]}']/seg", namespaces=nsmap).text:
                                segment_exists_in_other_language = True
                        elif tuv_name == "tuv lang" and \
                                tu.find(f"./tuv[@lang='{language_code_dict[l_inner]}']/seg", namespaces=nsmap) \
                                is not None:
                            if tu.find(f"./tuv[@lang='{language_code_dict[l_inner]}']/seg", namespaces=nsmap).text:
                                segment_exists_in_other_language = True
                    if segment_exists_in_other_language:
                        fps_texts_dict[languages_fps_dict[language.lower()]] += "None" + "\n"
                    else:
                        fps_texts_dict[languages_fps_dict[language.lower()]] += "\n"
    except etree.XMLSyntaxError:
        # Occurs when error parsing
        print(f"\n\nWARNING: error while processing tmx2txt!\n\n")

    # write output
    for out_fp, out_text in fps_texts_dict.items():
        with open(out_fp, "wt", encoding="utf-8") as out_f:
            out_f.write(out_text)


def prep_multilingual_ate_check(dp, languages, monolingual_ate):
    """
    Check parameters and paths for prep_multilingual_ate and return a dictionary
    with language codes as keys and paths to corpus directories of that language
    as values.

    :param dp: path to base directory where all data from corpus is stored
           with one or more .tmx files under the subdirectory "corpus"
    :param languages: list of languages ["en", "fr", "nl", "de"]
    :param monolingual_ate: type of monolingual ATE (currently only sbm supported)
    :return: main_dp (=path to main dir, ending in "/"), languages_dps_dict
    """
    if monolingual_ate != "sbm":
        raise ValueError(f"\nERROR prep_multilingual_ate: only 'sbm' monolingual ATE supported currently;\n"
                         f"not {monolingual_ate}")

    main_dp = check_corpus_dp(dp, ".tmx")  # check returns dp ending in "/"
    languages_dps_dict = {}
    for language in languages:  # for each language, make sure there is a separate main_dir for the monolingual ATE
        l = check_language(language)  # returns lowercased, 2-letter language code
        l_dp = main_dp[:-1] + f"_{l}/"  # create separate dir in same dir as main_dp per language
        l_dp_corpus = l_dp + "corpus/"  # create subdir "corpus" in that dp
        if not os.path.exists(l_dp):  # create language-dependent dir if it does not exist yet
            os.mkdir(l_dp)
        if not os.path.exists(l_dp_corpus):  # create language-dependent subdir corpus if it does not exist yet
            os.mkdir(l_dp_corpus)
        if monolingual_ate == "sbm":  # check for pre-existing data depending on chosen type of monolingual ATE
            proceed = check_existing_data_seq_no_features(l_dp)  # create data dp and check for pre-existing data
            if proceed == "stop":
                raise FileExistsError(f"ERROR prep_multilingual_ate: corpus has already been prepared/preprocessed;\n"
                                      f"Stopping prep_corpus_sbm for {main_dp}")
        languages_dps_dict[l] = l_dp_corpus
    return main_dp, languages_dps_dict


def prep_multilingual_ate(dp, languages, monolingual_ate="sbm"):
    """
    Prepare data for multilingual ATE, based on path to base directory,
    with one or more .tmx files  stored under the subdirectory "corpus":
    1- check parameters and paths
    2- extract .txt files from tmx and store in different directory
        > same names as original directory, but with "_[language]" added
    3- prep_corpus from appropriate monolingual_ate

    :param dp: path to base directory where all data from corpus is stored
           with one or more .tmx files under the subdirectory "corpus"
    :param languages: list of languages ["en", "fr", "nl", "de"]
    :param monolingual_ate: type of monolingual ATE (currently only sbm supported)
    :return: nothing but print progress and write files
    """
    print(f"\n\n##################################\n"
          f"# STARTING prep_multilingual_ate #\n"
          f"##################################\n\n")
    # 1. Check parameters and create output dirs per language
    main_dp, languages_dps_dict = prep_multilingual_ate_check(dp, languages, monolingual_ate)
    print(f"1. Checked parameters\n"
          f"\t* all ok, starting to prep for multilingual ATE with:\n"
          f"\t\t> path to main directory: {main_dp}\n"
          f"\t\t> languages: {languages}\n"
          f"\t\t> monolingual ATE: {monolingual_ate}\n\n"
          f"2. Extracting text from .tmx\n")

    # 2. Extract .txt files
    for l, l_dp_corpus in languages_dps_dict.items():
        print(f"\t\t> saving {l} txts in: {l_dp_corpus}\n")
    for tmx_fn in listdir_nohidden(main_dp + "corpus", ".tmx"):
        tmx_fp = main_dp + "corpus/" + tmx_fn
        tmx2txt(tmx_fp, languages_dps_dict)

    # 3. prep corpus
    if monolingual_ate == "sbm":
        print(f"3. Prepare corpus for sequential multilingual Bert monolingual term extraction\n")
        for l, l_dp_corpus in languages_dps_dict.items():
            l_dp = l_dp_corpus.replace("/corpus", "")
            prep_corpus_sbm(l_dp, l, tok_nesting="newlines")


def mono_sbm_ate_for_multilingual(dp, languages, domains, iob_or_io, optimiser="AdamW", nr_hidden=1, size=512,
                                  incl_incorr_tok=True, specific=1, common=1, ood=1, ne=1, partial=1):
    """
    Extract candidate terms from unseen corpus using pretrained models using the specified parameters,
    applied to all specified languages in the prepared multilingual (.tmx) corpus.

    :param dp: path to base directory where all data from corpus is stored
           with one or more .tmx files under the subdirectory "corpus"
    :param languages: list of languages in .tmx for which to perform this
    :param domains: list of domains to include ["corp", "equi", "htfl", "wind"]
    :param iob_or_io: use IOB or binary sequential IO labelling "io" or "iob"
    :param optimiser: optimiser to use "AdamW" (or "Adam")
    :param nr_hidden: number of hidden layers as integer (e.g., 1, 2, 3)
    :param size: size of hidden layers as integer (e.g., 128, 256, 512)
    :param incl_incorr_tok: whether to include partial annotations
            (of all labels indicated by following parameters)
    :param specific: whether to include Specific Terms
    :param common: whether to include Common Terms
    :param ood: whether to include OOD Terms
    :param ne: whether to include Named Entities
    :param partial: whether to include partial annotations of the previously defined labels
    :return: the paths to the output directories for l1 and l2
    """
    print(f"\n\n##########################################\n"
          f"# STARTING mono_sbm_ate_for_multilingual #\n"
          f"##########################################\n\n"
          f"* main directory of multilingual corpus: {dp}\n"
          f"* languages on which monolingual ATE will be performed: {languages}\n")
    main_dp = check_corpus_dp(dp, ".tmx")   # check path and return with "/"
    output_dps = []
    for language in languages:
        l = check_language(language)        # returns lowercased, 2-letter language code
        l_dp = main_dp[:-1] + f"_{l}/"
        output_dp = extract_terms_sbm(l_dp, domains, iob_or_io, optimiser=optimiser, nr_hidden=nr_hidden, size=size,
                                      incl_incorr_tok=incl_incorr_tok, specific=specific, common=common, ood=ood,
                                      ne=ne, partial=partial)
        output_dps.append(output_dp)
    return output_dps[0], output_dps[1]


def multilingual_ate_sbm_check(dp, l1, l2, l1_mono_output_dp, l2_mono_output_dp, out_fn):
    """
    Check the parameters and paths for multilingual_ate_sbm, returning
    2 dictionaries (one for each language) with relevant paths.

    :param dp: path to base directory where all data from corpus is stored
           with one or more .tmx files under the subdirectory "corpus"
    :param l1: first language ("en", "fr", "nl", or "de")
    :param l2: second language ("en", "fr", "nl", or "de")
    :param l1_mono_output_dp: path to dir where monolingual sbm ATE output is stored that should be used for l1
                              subdir of l2_main_dir/output_seq_bert_multi; can also be just name of dir and
                              the path will be automatically completed
    :param l2_mono_output_dp: path to dir where monolingual sbm ATE output is stored that should be used for l2
                              subdir of l2_main_dir/output_seq_bert_multi; can also be just name of dir and
                              the path will be automatically completed
    :param out_fn: name of file where output will be written (path will be automatically determined)
    :return: output_fp, l1_paths_dict, l2_paths_dict
    """
    main_dp = check_corpus_dp(dp, ".tmx")     # check returns dp ending in "/"
    if not l1_mono_output_dp.endswith("/"):   # make sure mono_output_dps end with "/"
        l1_mono_output_dp += "/"
    if not l2_mono_output_dp.endswith("/"):
        l2_mono_output_dp += "/"
    if l1 == l2:
        raise ValueError(f"\nERROR multilingual_ate_sbm_check: l1 and l2 are supposed to be different, not both {l1}\n")

    out_dp = main_dp + "output_multilingual/"
    if not os.path.exists(out_dp):
        os.mkdir(out_dp)
    if "." not in out_fn:
        out_fn += ".tsv"
    if "/" in out_fn:
        raise ValueError(f"\nERROR multilingual_ate_sbm_check: out_fn is supposed to be a filename, not path: {out_fn}")
    out_fp = out_dp + out_fn

    l1_paths_dict = {}                       # get dicts per language with all relevant paths
    l2_paths_dict = {}
    for l in [l1, l2]:
        l = check_language(l)                                   # check language lowercased and with 2-letter code
        l_dp = main_dp[:-1] + f"_{l}/"                          # get main dir per language
        l_tok_corpus_dp = l_dp + "data_seq_no_features/"        # get tokenised corpus subdir per language
        l_main_output_dp = l_dp + "output_seq_bert_multi/"      # get main dir for monolingual sbm output per language

        # if only dir names are given of monolingual sbm ATE output, turn them into full paths
        if l == l1 and not l1_mono_output_dp.startswith(l_main_output_dp):
            l1_mono_output_dp = l_main_output_dp + l1_mono_output_dp
            if not os.path.exists(l1_mono_output_dp):
                raise ValueError(f"\nERROR multilingual_ate_sbm_check: directory was expected but does not exist: \n"
                                 f"{l1_mono_output_dp}\n"
                                 f"make sure to prep_multilingual_ate and mono_sbm_ate_for_multilingual before "
                                 f"running multilingual_ate_sbm!\n")
        if l == l2 and not l2_mono_output_dp.startswith(l_main_output_dp):
            l2_mono_output_dp = l_main_output_dp + l2_mono_output_dp
            if not os.path.exists(l2_mono_output_dp):
                raise ValueError(f"\nERROR multilingual_ate_sbm_check: directory was expected but does not exist: \n"
                                 f"{l2_mono_output_dp}\n"
                                 f"make sure to prep_multilingual_ate and mono_sbm_ate_for_multilingual before "
                                 f"running multilingual_ate_sbm!\n")

        # check whether all paths exist
        for dp in [l_dp, l_tok_corpus_dp, l_main_output_dp]:
            if not os.path.exists(dp):
                raise ValueError(f"\nERROR multilingual_ate_sbm_check: directory was expected but does not exist: \n"
                                 f"{dp}\n"
                                 f"make sure to prep_multilingual_ate and mono_sbm_ate_for_multilingual before "
                                 f"running multilingual_ate_sbm!\n")

        # add paths to dicts
        if l == l1:
            l1_paths_dict["main_dp"] = l_dp
            l1_paths_dict["tok_corpus_dp"] = l_tok_corpus_dp
            l1_paths_dict["main_output_dp"] = l_main_output_dp
            l1_paths_dict["mono_output_dp"] = l1_mono_output_dp
            l1_paths_dict["mono_output_termlist_fp"] = l1_mono_output_dp + "combined_termlist.tsv"
        elif l == l2:
            l2_paths_dict["main_dp"] = l_dp
            l2_paths_dict["tok_corpus_dp"] = l_tok_corpus_dp
            l2_paths_dict["main_output_dp"] = l_main_output_dp
            l2_paths_dict["mono_output_dp"] = l2_mono_output_dp
            l2_paths_dict["mono_output_termlist_fp"] = l2_mono_output_dp + "combined_termlist.tsv"

    return out_fp, l1_paths_dict, l2_paths_dict


def tok_corpus_dp_to_dicts(tok_corpus_dp):
    """
    Based on the path to a directory where a tokenised corpus is saved (data_seq_no_features),
    create 3 dictionaries with file ids (filename without language code and extension) as keys
    (these IDs should be the same in different languages of a parallel corpus) and as values:
    1) nested list of tokens per sentence
    2) list of all tokens with empty token between sentences
    3) string of all tokens with space between tokens and "*_*" between sentences

    :param tok_corpus_dp: path to a directory where a tokenised corpus is saved (data_seq_no_features)
    :return: txts_nested_list_dict, texts_list_dict, texts_string_dict
    """
    txts_list_dict = {}
    txts_string_dict = {}
    txts_nested_list_dict = {}
    for txt_fn in listdir_nohidden(tok_corpus_dp, extension=".txt"):
        txt_fp = tok_corpus_dp + txt_fn
        txt_id = txt_fn[:-7]
        with open(txt_fp, "rt", encoding="utf-8") as txt_f:
            txts_list_dict[txt_id] = txt_f.read().splitlines()
            list_with_special_char_for_eos = ["*_*" if x == "" else x for x in txts_list_dict[txt_id]]
            text = " ".join(list_with_special_char_for_eos)
            nested_list = [[]]
            for token in list_with_special_char_for_eos:
                if token != "*_*":
                    nested_list[-1].append(token)
                else:
                    if nested_list[-1]:
                        nested_list.append([])
                    else:
                        nested_list.append([])

            txts_string_dict[txt_id] = text
            txts_nested_list_dict[txt_id] = nested_list
    return txts_nested_list_dict, txts_list_dict, txts_string_dict


def align_sentences(l1_tok_sentences, l2_tok_sentences):
    """
    Given 2 nested lists of tokenised sentences, in 2 languages (sentence-aligned)
    use the ASTrED word aligner to get the alignments.

    :param l1_tok_sentences: nested list of tokenised sentences in l1
    :param l2_tok_sentences: nested list of tokenised sentences in l2
    :return: alignments in 2 forms:
             alignments_tuples = e.g., [[(0, 0), (1, 1), (2, 2), (2, 3), (3, 4), (4, 5)], [], ...]
             alignment_dicts = e.g, [{0: 0, 1: 1, 2: 2, 2: 3, ...}, {}, ...]
    """
    alignments_tuples = []
    alignments_dicts = []
    aligner = Aligner()
    if len(l1_tok_sentences) != len(l2_tok_sentences):
        print(F"ERROR: {len(l1_tok_sentences)} {len(l2_tok_sentences)}")
    for l1_sentence, l2_sentence in zip(l1_tok_sentences, l2_tok_sentences):
        sentence_alignments = aligner.align(" ".join(l1_sentence), " ".join(l2_sentence))
        alignments_tuples.append(sentence_alignments)
        alignment_dict = {}
        for token_alignment in sentence_alignments:
            alignment_dict[token_alignment[0]] = token_alignment[1]
        alignments_dicts.append(alignment_dict)
    return alignments_tuples, alignments_dicts


def get_cts_from_combined_termlist(combined_termlist_fp):
    """
    Based on the path to a combined_termlist.tsv file, get a list of all candidate terms.

    :param combined_termlist_fp: path to a combined_termlist.tsv file
    :return: [candidate terms]
    """
    cts = []
    with open(combined_termlist_fp, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            ct = line.split("\t")[0]
            cts.append(ct)
    return cts


def get_ct_indices(ct_list, txts_nested_list_dict, txts_string_dict):
    """
    Based on a list of (tokenised, but not with list, simply with spaces between tokens) candidate terms,
    and two dictionaries with txt_ids as keys and as values:
    1) nested list of sentences and tokens
    2) single string of all (tokenised) text
    Derive a triple-nested dictionary with candidate terms as keys,
        txt_ids as keys
            sentence indices as keys
                list of tuples of start and end token indices for the ct [(token_id_start, token_id_end), (), ...]

    :param ct_list: list of (tokenised, but not with list, simply with spaces between tokens) candidate terms,
    :param txts_nested_list_dict: dictionary with txt_ids as keys and as values nested list of sentences and tokens
    :param txts_string_dict: dictionary with txt_ids as keys and as values single string of all (tokenised) text
    :return: ct_indices_dict {ct: {txt_id: {sentence_i: [(token_id_start, token_id_end), (), ...]}, ...}, ...}
             cts_per_sentence_dict {txt_id: {sentence_i: [cts], ...}, ...}
    """
    ct_indices_dict = {}        # {ct: {txt_id: {sentence_i: [[indices of tokens in 1st occurrence], ...]}, ...}, ...}
    cts_per_sentence_dict = {}  # {txt_id: {sentence_i: [cts], ...}, ...}
    for ct in ct_list:
        ct_indices_dict[ct] = {}
        for txt_id, txt_string in txts_string_dict.items():
            if ct.lower() in txt_string.lower():
                nested_tokenised_txt = txts_nested_list_dict[txt_id]
                if txt_id not in cts_per_sentence_dict:
                    cts_per_sentence_dict[txt_id] = {}
                for sentence_i, sentence_list in enumerate(nested_tokenised_txt):
                    ct_lower_list = ct.lower().split(" ")
                    sentence_list_lower = [token.lower() for token in sentence_list]
                    sublist_indices = get_sublist_indices(ct_lower_list, sentence_list_lower)
                    if sublist_indices:
                        if txt_id not in ct_indices_dict[ct]:
                            ct_indices_dict[ct][txt_id] = {}
                        ct_indices_dict[ct][txt_id][sentence_i] = sublist_indices
                        if sentence_i not in cts_per_sentence_dict[txt_id]:
                            cts_per_sentence_dict[txt_id][sentence_i] = [ct]
                        else:
                            cts_per_sentence_dict[txt_id][sentence_i].append(ct)
    return ct_indices_dict, cts_per_sentence_dict


def align_candidate_terms(l1_ct_indices_dict, l2_ct_indices_dict, l2_cts_per_sentence_dict, alignments_dict):
    """
    Based on data obtained in multilingual_ate_sbm, create an alignment dictionary where
    candidate terms from L1 are aligned to one or more candidate terms from L2 based
    on the extracted lists of monolingual candidate terms and the word alignments.

    :param l1_ct_indices_dict: {ct: {txt_id: {sentence_i: [[indices of tokens in 1st occurrence], ...]}, ...}, ...}
    :param l2_ct_indices_dict: {ct: {txt_id: {sentence_i: [[indices of tokens in 1st occurrence], ...]}, ...}, ...}
    :param l2_cts_per_sentence_dict: {txt_id: {sentence_i: [cts], ...}, ...}
    :param alignments_dict: {txt_id: [{0: 0, 1: 1, 2: 2, 2: 3, ...}, {}, ...]}
    :return: l1_l2_ct_alignment_dict = {l1_ct: {l2_ct: {txt_id: {sentence_i: [alignment_percentages], ...}, ...}, ...}}
             l1_ct_freqdict = {l1_ct: freq, ...}
    """
    l1_l2_ct_alignment_dict = {}
    l1_ct_freqdict = {}
    for l1_ct, l1_txt_sentences_dict in l1_ct_indices_dict.items():
        l1_ct_freq = 0
        l1_l2_ct_alignment_dict[l1_ct] = {}
        for txt_id, l1_sentence_token_indices_dict in l1_txt_sentences_dict.items():
            for sentence_i, l1_token_indices in l1_sentence_token_indices_dict.items():
                if sentence_i in l2_cts_per_sentence_dict[txt_id]:
                    l2_cts_in_sentence = l2_cts_per_sentence_dict[txt_id][sentence_i]
                else:
                    l2_cts_in_sentence = []
                for l1_ct_indices_of_occurrence in l1_token_indices:
                    l1_ct_freq += 1
                    target_alignments = []
                    sentence_alignment_dict = alignments_dict[txt_id][sentence_i]
                    for token_i in l1_ct_indices_of_occurrence:
                        if token_i in sentence_alignment_dict:
                            target_i = sentence_alignment_dict[token_i]
                            target_alignments.append(target_i)
                    for l2_ct in l2_cts_in_sentence:
                        l2_ct_indices_of_occurrences = l2_ct_indices_dict[l2_ct][txt_id][sentence_i]
                        for l2_ct_indices_of_occurrence in l2_ct_indices_of_occurrences:
                            overlap = len(set(l2_ct_indices_of_occurrence).intersection(target_alignments)) / \
                                      max(len(set(l2_ct_indices_of_occurrence)), len(set(target_alignments)))
                            if l2_ct not in l1_l2_ct_alignment_dict[l1_ct]:
                                l1_l2_ct_alignment_dict[l1_ct][l2_ct] = []
                            l1_l2_ct_alignment_dict[l1_ct][l2_ct].append(overlap)
        l1_ct_freqdict[l1_ct] = l1_ct_freq
    return l1_l2_ct_alignment_dict, l1_ct_freqdict


def write_alignments(l1_l2_ct_alignment_dict, l1_ct_freqdict, out_fp, threshold=0.5):
    """
    Based on the alignment dict from align_candidate_terms,
    write the output in an ordered way to a file with the given path

    :param l1_l2_ct_alignment_dict: {l1_ct: {l2_ct: {txt_id: {sentence_i: [alignment_percentages], ...}, ...}, ...}}
    :param l1_ct_freqdict: frequency dictionary of l1 candidate terms
    :param out_fp: path to file where output will be written
    :param threshold: threshold value to include potential equivalents
    :return: nothing
    """
    results = {}
    for l1_ct, l2_ct_candidates_dict in l1_l2_ct_alignment_dict.items():
        l2_ct_score = {}
        l2_ct_scores_as_strings = {}
        l1_ct_freq = l1_ct_freqdict[l1_ct]
        for l2_ct, alignment_matches in l2_ct_candidates_dict.items():
            times_in_same_sentence = len(alignment_matches)
            perc_in_same_sentence = times_in_same_sentence / l1_ct_freq
            average_score = statistics.mean(alignment_matches)
            nr_full_matches = alignment_matches.count(1.0)
            perc_full_matches = nr_full_matches / l1_ct_freq
            total_score = (perc_full_matches * 2) + (average_score * 2) + perc_in_same_sentence
            l2_ct_score[l2_ct] = total_score
            l2_ct_scores_as_strings[l2_ct] = [str(perc_in_same_sentence), str(average_score), str(nr_full_matches),
                                              str(perc_full_matches), str(total_score)]
        results[l1_ct] = {}
        for l2_ct, final_score in sorted(l2_ct_score.items(), key=itemgetter(1), reverse=True):
            if final_score < threshold and results[l1_ct]:
                break
            results[l1_ct][l2_ct] = l2_ct_scores_as_strings[l2_ct]

    to_write = [["L1 Candidate Term",
                 "Potentially Equivalent L2 Candidate Term",
                 "A: Occurrences in same sentence/L1 CT occurrences",
                 "B: Average word alignment match percentage",
                 "C: Total number of full matches",
                 "D: Number of full matches/L1 CT occurrences",
                 "E: Combined score (A + 2B + 2D)"]]
    for l1_ct, l2_ct_scores_dict in results.items():
        lines_to_write = []
        for l2_ct, l2_ct_scores in l2_ct_scores_dict.items():
            if not lines_to_write:
                lines_to_write.append([l1_ct, l2_ct])
                lines_to_write[-1] += l2_ct_scores
            else:
                lines_to_write.append([" ", l2_ct])
                lines_to_write[-1] += l2_ct_scores
        for line in lines_to_write:
            to_write.append(line)
    with open(out_fp, "wt", encoding="utf-8") as out_f:
        for line in to_write:
            out_f.write("\t".join(line) + "\n")


def multilingual_ate_sbm(dp, l1, l2, l1_mono_output_dp, l2_mono_output_dp, out_fn):
    """
    Based on the output of monolingual term extraction (the sequential, bert-multilingual version),
    align the candidate terms crosslingually using ASTrED word alignments
    and frequency ratios. Write the output to the specified file.

    :param dp: path to base directory where all data from corpus is stored
           with one or more .tmx files under the subdirectory "corpus"
    :param l1: first language ("en", "fr", "nl", or "de")
    :param l2: second language ("en", "fr", "nl", or "de")
    :param l1_mono_output_dp: path to dir where monolingual sbm ATE output is stored that should be used for l1
                              subdir of l2_main_dir/output_seq_bert_multi; can also be just name of dir and
                              the path will be automatically completed
    :param l2_mono_output_dp: path to dir where monolingual sbm ATE output is stored that should be used for l2
                              subdir of l2_main_dir/output_seq_bert_multi; can also be just name of dir and
                              the path will be automatically completed
    :param out_fn: name of file where output will be written (path will be automatically determined)
    :return: nothing, but print progress
    """
    print(f"\n\n#################################\n"
          f"# STARTING multilingual_ate_sbm #\n"
          f"#################################\n\n"
          f"1. Checking parameters and paths")
    # 1. check parameters and paths
    out_fp, l1_paths_dict, l2_paths_dict = multilingual_ate_sbm_check(dp, l1, l2, l1_mono_output_dp,
                                                                      l2_mono_output_dp, out_fn)
    print(f"\t> got all relevant paths\n"
          f"\t> L1 paths: {l1_paths_dict}\n"
          f"\t> L2 paths: {l2_paths_dict}\n"
          f"\t> output filepath: {out_fp}\n"
          f"2. Getting tokenised data")

    # 2. get dicts of all tokenised sentences per text with language-independent file_ids as keys and as values:
    # nested list of tokenised sentences, single list of tokens, single tokenised string
    l1_txts_nested_list_dict, l1_txts_list_dict, l1_txts_string_dict = \
        tok_corpus_dp_to_dicts(l1_paths_dict["tok_corpus_dp"])
    l2_txts_nested_list_dict, l2_txts_list_dict, l2_txts_string_dict = \
        tok_corpus_dp_to_dicts(l2_paths_dict["tok_corpus_dp"])
    print("\t> got tokenised data\n"
          "3. Getting word alignments")

    # 3. get alignments per text {txt_id: [{0: 0, 1: 1, 2: 2, 2: 3, ...}, {}, ...]}
    alignments_txt_dicts = {}
    for txt_id, l1_tok_sentences in l1_txts_nested_list_dict.items():
        l2_tok_sentences = l2_txts_nested_list_dict[txt_id]
        alignment_tuples, alignment_dicts = align_sentences(l1_tok_sentences, l2_tok_sentences)
        alignments_txt_dicts[txt_id] = alignment_dicts
    print("\t> got alignments\n"
          "4. Getting candidate terms")

    # 4. get candidate terms
    cts_l1 = get_cts_from_combined_termlist(l1_paths_dict["mono_output_termlist_fp"])
    cts_l2 = get_cts_from_combined_termlist(l2_paths_dict["mono_output_termlist_fp"])
    print("\t> got candidate terms\n"
          "5. Getting indices of candidate terms to match them with word alignments")

    # 5. get indices of cts and cts per sentence:
    # ct_indices_dict {ct: {txt_id: {sentence_i: [(token_id_start, token_id_end), (), ...]}, ...}, ...}
    # cts_per_sentence_dict {txt_id: {sentence_i: [cts], ...}, ...}
    l1_ct_indices_dict, l1_cts_per_sentence_dict = get_ct_indices(cts_l1, l1_txts_nested_list_dict, l1_txts_string_dict)
    l2_ct_indices_dict, l2_cts_per_sentence_dict = get_ct_indices(cts_l2, l2_txts_nested_list_dict, l2_txts_string_dict)
    print("\t> got indices\n"
          "6. Aligning candidate terms")

    # 6. align candidate terms
    l1_l2_ct_alignment_dict, l1_ct_freqdict = align_candidate_terms(l1_ct_indices_dict, l2_ct_indices_dict,
                                                                    l2_cts_per_sentence_dict, alignments_txt_dicts)
    print("\t> aligned candidate terms\n"
          "7. Writing output")

    # 7. write output
    write_alignments(l1_l2_ct_alignment_dict, l1_ct_freqdict, out_fp)
    print(f"\t> wrote output to {out_fp}")


def multilingual_ate_sbm_complete(dp, out_fn, l1, l2, domains, iob_or_io, optimiser="AdamW", nr_hidden=1, size=512,
                                  incl_incorr_tok=True, specific=1, common=1, ood=1, ne=1, partial=1):
    """
    Full pipeline for multilingual ATE based on monolingual (sequential Bert-multilingual) ATE, with:
        1) preparation through prep_multilingual_ate
        2) monolingual term extraction with mono_sbm_ate_for_multilingual
        3) cross-lingual alignment with multilingual_ate_sbm
    Separate directories will be created to store the results of the monolingual extractions and
    the results of the alignments will be stored in the project directory under an "output" subdirectory
    with the given filename

    :param dp: path to base directory where all data from corpus is stored
           with one or more .tmx files under the subdirectory "corpus"
    :param out_fn: name of file where output will be written (path will be automatically determined)
    :param l1: first language ("en", "fr", "nl", or "de")
    :param l2: second language ("en", "fr", "nl", or "de")
    :param domains: list of domains to include ["corp", "equi", "htfl", "wind"]
    :param iob_or_io: use IOB or binary sequential IO labelling "io" or "iob"
    :param optimiser: optimiser to use "AdamW" (or "Adam")
    :param nr_hidden: number of hidden layers as integer (e.g., 1, 2, 3)
    :param size: size of hidden layers as integer (e.g., 128, 256, 512)
    :param incl_incorr_tok: whether to include partial annotations
            (of all labels indicated by following parameters)
    :param specific: whether to include Specific Terms
    :param common: whether to include Common Terms
    :param ood: whether to include OOD Terms
    :param ne: whether to include Named Entities
    :param partial: whether to include partial annotations of the previously defined labels
    :return: nothing, but print progress and write results
    """
    languages = [l1, l2]
    prep_multilingual_ate(dp, languages, monolingual_ate="sbm")
    output_dp_l1, output_dp_l2 = mono_sbm_ate_for_multilingual(dp, languages, domains, iob_or_io, optimiser=optimiser,
                                                               nr_hidden=nr_hidden, size=size,
                                                               incl_incorr_tok=incl_incorr_tok,
                                                               specific=specific, common=common, ood=ood, ne=ne,
                                                               partial=partial)
    multilingual_ate_sbm(dp, l1, l2, output_dp_l1, output_dp_l2, out_fn)
