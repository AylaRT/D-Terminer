"""
Script for the pipeline of sequential automatic term extraction, using a recurrent neural network
through the Flair framework and multilingual Bert word embeddings. The models are pretrained using a
script in a different repository: ID-TERM/SaMLET_train_and_save_models.py

This specific pipeline (sequential, using multilingual BERT embeddings, i.e., seq_bert_multi),
will consistently be abbreviated as "sbm", making it easier to have informative, yet unique function names.

Author: Ayla Rigouts Terryn
Created: 18/03/2022
Last updated: 04/05/2022
"""


import os
from flair.data import Sentence
from flair.models import SequenceTagger
from prepro_linguistic import tokenise_text
from prepro_normalise import normalise_text_standard
from dterminer_reusables import check_language, check_corpus_dp, check_existing_data_seq_no_features, listdir_nohidden


def prep_corpus_sbm(dp, language, tok_nesting="eos"):
    """
    Preprocess/prepare the unseen corpus in preparation of the sbm term extraction.

    Prerequisites: The directory is expected to be subdirectory of "D-Terminer/unseen_corpora",
    with a main directory where all info on the corpus will be stored, i.e.: "D-Terminer/unseen_corpora/[corpus_name]/"
    (the given path (dp) should point to this main directory)! The corpus itself should be stored in a subdirectory
    called "corpus", i.e.: "D-Terminer/unseen_corpora/[corpus_name]/corpus/".
    Different subdirectories will be created to store the preprocessed corpus and the results.
    With prep_corpus_sbm, only "data_seq_no_features" is created.

    Preprocessing/preparation includes:
    1. minor normalisation > see normalise.py
    2. tokenisation with LeTs Preprocess (incl. EOS)
    3. write tokenised results to separate file under new, empty directory "corpus_prepro_sbm",
       with one token per line and an empty line between sentences.

    :param dp: path to main directory where corpus is stored under subdirectory "corpus"
    :param language: "nl", "fr", "en", or "de"
    :param tok_nesting: how tokenisation will be performed
                    "eos" => nested for each sentence
                    "newlines" => only nesting for newlines in original text
                    "none" => no nesting
    :return: nothing, but print progress + write preprocessed corpus to data_seq_no_features directory,
                                           with same filenames as originals, but starting with "seqData_"
    """
    print(f"\n\n############################\n"
          f"# STARTING prep_corpus_sbm #\n"
          f"############################\n\n")
    # 1. Check parameters
    l = check_language(language)    # check returns l as lowercased and abbreviated form
    main_dp = check_corpus_dp(dp, ".txt")   # check returns dp ending in "/"
    proceed = check_existing_data_seq_no_features(main_dp)    # create data dp and check for pre-existing data
    if proceed == "stop":
        raise FileExistsError(f"ERROR prep_corpus_sbm: corpus has already been prepared/preprocessed;\n"
                              f"Stopping prep_corpus_sbm for {main_dp}")
    print(f"1. Checked parameters\n"
          f"\t* both ok, so proceeding to normalisation of corpus with:\n"
          f"\t\t> path to main directory: {main_dp}\n"
          f"\t\t> language: {l}\n\n"
          f"2. Starting normalisation, tokenisation, and writing output\n"
          f"\t* output dp {main_dp + 'data_seq_no_features/'}")

    # 2. Per file: normalise, tokenise, write output to data_seq_no_features directory
    corpus_dp = main_dp + "corpus/"
    data_dp = main_dp + "data_seq_no_features/"
    for fn in listdir_nohidden(corpus_dp, extension=".txt"):
        # 2.1 get filepaths of original txt file and file where sequential data will be written
        corpus_fp = corpus_dp + fn
        fn_data = "seqData_" + fn
        data_fp = data_dp + fn_data

        # 2.2 normalise
        # result = single string of normalised text (can be multiple lines)
        with open(corpus_fp, "rt", encoding="utf-8") as corpus_f:
            original_text = corpus_f.read()
        normalised_text = normalise_text_standard(original_text)

        # 2.3 tokenise
        # result = nested list of sentences and tokens, based on normalised text
        tokenised_sentences = tokenise_text(normalised_text, l, nesting=tok_nesting)

        # 2.4 write output
        # result = file with one token per line and empty line between sentences
        to_write = ""
        for sentence in tokenised_sentences:
            for token in sentence:
                to_write += token + "\n"
            to_write += "\n"
        with open(data_fp, "wt", encoding="utf-8") as data_f:
            data_f.write(to_write)
        print(f"\t\t> done with {data_fp}")
    print("\n\nCOMPLETED prep_corpus_sbm\n")


def check_extract_terms_sbm_parameters(domains, iob_or_io, optimiser, nr_hidden, size,
                                       incl_incorr_tok, specific, common, ood, ne, partial):
    """
    Check parameters for extract_terms_sbm

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
    :return: nothing; raise error if necessary
    """
    # check domains
    possible_domains = ["corp", "equi", "htfl", "wind"]
    for domain in domains:
        if domain not in possible_domains:
            raise ValueError(f"ERROR extract_terms_sbm: unsupported domain ({domain});\n"
                             f"the only domains are: 'corp', 'equi', 'htfl', or 'wind'")

    # check iob_or_io
    if iob_or_io.lower() not in ["iob", "io"]:
        raise ValueError(f"ERROR extract_terms_sbm: unexpected value for iob_or_io parameter: {iob_or_io}")

    # check optimiser, nr_hidden, size
    if optimiser not in ["Adam", "AdamW"]:
        raise ValueError(f"ERROR extract_terms_sbm: optimiser not recognised: {optimiser};\n"
                         f"script currently only supports 'Adam' and 'AdamW'.")
    if nr_hidden not in list(range(1, 11)):
        raise ValueError(f"ERROR extract_terms_sbm: unexpected value for number of hidden layers: {nr_hidden};\n"
                         f"expected a value between 1 and 10 (included).")
    if size not in [128, 256, 512, 1024, 2048]:
        raise ValueError(f"ERROR extract_terms_sbm: unexpected value for size of hidden layers: {size};\n"
                         f"expected one of the following values: [128, 256, 512, 1024, 2048].")

    # check labels
    for label_name, label_value in {"specific": specific, "common": common, "ood": ood, "ne": ne,
                                    "partial": partial}.items():
        if label_value != 1 and label_value != 0:
            raise ValueError(f"ERROR extract_terms_sbm: label {label_name} is {label_value} instead of 1 or 0")
    if type(incl_incorr_tok) != bool:
        raise ValueError(f"ERROR extract_terms_sbm: incl_incorr_tok is supposed to be boolean, not "
                         f"{incl_incorr_tok}")


def check_sbm_pretrained_model(domains, iob_or_io, optimiser, nr_hidden, size,
                               incl_incorr_tok, specific, common, ood, ne, partial):
    """
    Check whether the pretrained model is available to perform sbm term extraction with the specified parameters.

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
    :return: model_fp (path to file where model is stored)
    """
    model_dp = "pretrained_models/seq_bert_multi/"

    # check whether optimiser/nr_hidden/size/inclIncorrTok are standard settings
    if optimiser == "AdamW" and nr_hidden == 1 and size == 512 and incl_incorr_tok:
        settings_string = "standardSettings"
    else:
        settings_string = f"opt{optimiser}_hidden{nr_hidden}_size{size}_"
        if incl_incorr_tok:
            settings_string += "inclIncorrTok"
        else:
            settings_string += "exclIncorrTok"
    # check domains
    if domains == ["corp", "equi", "htfl", "wind"]:
        domain_string = "allDomains"
    else:
        domain_string = ""
        for domain in domains:
            if domain_string:
                domain_string += domain.title()
            else:
                domain_string += domain
    # check labels
    label_string = str(specific) + str(common) + str(ood) + str(ne) + str(partial)

    # get model_fp and check if available
    model_fn = domain_string + "_" + iob_or_io.lower() + "_" + label_string + "_" + settings_string + ".pt"
    model_fp = model_dp + model_fn
    if not os.path.exists(model_fp):
        raise ValueError(f"ERROR extract_terms_sbm: the pretrained model with the specified settings is currently "
                         f"unavailable, i.e., {model_fp} does not exist!")
    return model_fp


def check_sbm_data(main_dp):
    """
    Check whether data has been prepared correctly for the corpus in the given directory.
    So, check whether there is a subdir "data_seq_no_features" under the main dir,
    that has the same number of txt files as the "corpus" subdirectory.

    :param main_dp: path to main directory where corpus is stored under subdirectory "corpus"
    :return: data_dp (path to directory where relevant data for sbm term extraction is stored)
    """
    data_dp = main_dp + "data_seq_no_features/"
    corpus_dp = main_dp + "corpus/"
    if not os.path.exists(data_dp):
        raise ValueError(f"ERROR extract_terms_sbm: directory where prepared data is supposed to be stored "
                         f"does not exist:\n{data_dp}")
    nr_corpus_files = len(list(listdir_nohidden(corpus_dp, ".txt")))
    nr_data_files = len(list(listdir_nohidden(data_dp, ".txt")))
    if nr_corpus_files != nr_data_files:
        raise ValueError(f"ERROR extract_terms_sbm: different number of txt-files in corpus versus prepared data:\n"
                         f"{corpus_dp}: {nr_corpus_files} files;\n"
                         f"{data_dp}: {nr_data_files} files;")
    return data_dp


def get_sbm_output_dp(main_dp, model_fp):
    """
    Based on the path to the main project directory for the corpus and the path to the pretrained model,
    create and return the output dp for the sbm term extraction.

    :param main_dp: path to main directory where corpus is stored under subdirectory "corpus"
    :param model_fp: path to file where pretrained model is stored
    :return: output_dp
    """
    # create main directory where subdir for current experiment will be stored
    main_output_dp = main_dp + "output_seq_bert_multi/"
    if not os.path.exists(main_output_dp):
        os.mkdir(main_output_dp)
    # get specific output dp based on model dp
    output_dp = model_fp.replace(".pt", "_output/")
    output_dp = output_dp.replace("pretrained_models/seq_bert_multi/", main_output_dp)
    if not os.path.exists(output_dp):
        os.mkdir(output_dp)
    # check if output files already exist and ask if they should be overwritten
    if len(list(listdir_nohidden(output_dp, extension=".tsv"))) > 0:
        proceed = ""
        while proceed not in ["stop", "overwrite"]:
            proceed = input(f"INPUT REQUESTED extract_terms_sbm > get_sbm_output_dp:\n"
                            f"the directory {output_dp} \n"
                            f"already exists and appears to contain files.\n"
                            f"Do you want to remove these files and get output from scratch again?\n"
                            f"Or do you want to stop the term extraction process?\n"
                            f"Type 'overwrite' or 'stop' to choose.")
        if proceed == "overwrite":
            for fn in listdir_nohidden(output_dp):
                fp = output_dp + fn
                os.remove(fp)
        else:
            raise ValueError(f"STOPPED extract_terms_sbm because of pre-existing output")
    return output_dp


def get_data_sbm(data_dp):
    """
    Get the prepared data (see prep_corpus_sbm) for extract_terms_sbm as a dictionary with a nested list:
    data_dict = {filename: [[token1_sentence1, token2_sentence1, ...], [token1_sentence2, token2_sentence2, ...], ...]}
    as filename for the keys, use the filename of the output file, which is seqOutput_[original_filename].tsv

    :param data_dp: path to directory where prepared data is stored in separate files
    :return: data_dict
    """
    data_dict = {}
    for data_fn in listdir_nohidden(data_dp, extension=".txt"):
        output_fn = data_fn.replace("seqData_", "seqOutput_")
        output_fn = output_fn.replace(".txt", ".tsv")
        data_fp = data_dp + data_fn
        data_dict[output_fn] = [[]]
        with open(data_fp, "rt", encoding="utf-8") as data_f:
            for line in data_f.readlines():
                line = line.strip()
                if line:
                    data_dict[output_fn][-1].append(line)
                else:
                    data_dict[output_fn].append([])
    return data_dict


def map_capitalisation_termlist(candidate_term_dict):
    """
    Given the candidate_term_dict from extract_terms_sbm, i.e., a dictionary with
    {ct_original_caps: [list of all files in which ct occurs, one filename for each occurrence, so including doubles],
    ...}
    combine all cts which are identical apart from the capitalisation and create a new
    dictionary with the same structure, in which the keys are those cts with the fewest
    uppercase letters out of all occurrences in the corpus

    :param candidate_term_dict: {ct_original_caps: [list of all files in which ct occurs,], ...}
    :return: capscorrected_candidate_term_dict
    """
    capscorrected_candidate_term_dict = {}
    all_cts = list(candidate_term_dict.keys())
    all_cts_lower = []
    for ct in all_cts:
        all_cts_lower.append(ct.lower())
    for ct, ct_files in candidate_term_dict.items():
        # if lowercased ct exists:
        if ct == ct.lower() or ct.lower() in candidate_term_dict:
            if ct.lower() not in capscorrected_candidate_term_dict:
                capscorrected_candidate_term_dict[ct.lower()] = ct_files
            else:
                capscorrected_candidate_term_dict[ct.lower()] += ct_files
        # if lowercased ct does not exist
        else:
            nr_occurrences_ct = all_cts_lower.count(ct.lower())
            if nr_occurrences_ct == 1:
                capscorrected_candidate_term_dict[ct] = ct_files
            elif ct not in [x.lower() for x in candidate_term_dict.keys()] and \
                    ct.lower() not in [x.lower() for x in capscorrected_candidate_term_dict.keys()]:
                ct_least_caps = ct
                ct_least_cap_freq = len(ct_files)
                least_nr_caps_ct = sum(1 for c in ct if c.isupper())
                combined_filelist = ct_files
                for ct_inner_loop, files_inner_loop in candidate_term_dict.items():
                    if ct.lower() == ct_inner_loop.lower() and ct_inner_loop != ct:
                        combined_filelist += files_inner_loop
                        nr_caps_ct_inner_loop = sum(1 for c in ct_inner_loop if c.isupper())
                        if nr_caps_ct_inner_loop < least_nr_caps_ct:
                            ct_least_caps = ct_inner_loop
                            least_nr_caps_ct = nr_caps_ct_inner_loop
                            ct_least_cap_freq = len(files_inner_loop)
                        elif nr_caps_ct_inner_loop == least_nr_caps_ct:
                            if len(files_inner_loop) > ct_least_cap_freq:
                                ct_least_caps = ct_inner_loop
                                least_nr_caps_ct = nr_caps_ct_inner_loop
                                ct_least_cap_freq = len(files_inner_loop)
                capscorrected_candidate_term_dict[ct_least_caps] = combined_filelist
    return capscorrected_candidate_term_dict


def extract_terms_sbm(dp, domains, iob_or_io, optimiser="AdamW", nr_hidden=1, size=512,
                      incl_incorr_tok=True, specific=1, common=1, ood=1, ne=1, partial=1):
    """
    Extract candidate terms from unseen corpus using pretrained models using the specified parameters.

    :param dp: path to main directory where corpus is stored under subdirectory "unseen_corpora"
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
    :return: path to output directory
    """
    print(f"\n\n##############################\n"
          f"# STARTING extract_terms_sbm #\n"
          f"##############################\n\n")
    # 1. Check parameters & necessary files + get relevant paths
    main_dp = check_corpus_dp(dp, ".txt")
    check_extract_terms_sbm_parameters(domains, iob_or_io, optimiser, nr_hidden, size,
                                       incl_incorr_tok, specific, common, ood, ne, partial)
    data_dp = check_sbm_data(main_dp)
    model_fp = check_sbm_pretrained_model(domains, iob_or_io, optimiser, nr_hidden, size,
                                          incl_incorr_tok, specific, common, ood, ne, partial)
    output_dp = get_sbm_output_dp(main_dp, model_fp)
    print(f"1. Checked parameters\n"
          f"\t* Everything ok, starting term extraction with:\n"
          f"\t\t> main directory: {main_dp}\n"
          f"\t\t> data directory: {data_dp}\n"
          f"\t\t> saved model: {model_fp}\n"
          f"\t\t> output directory: {output_dp}\n"
          f"2. Starting extraction and writing output\n"
          f"\t* Getting data\n")

    # 2. Get data and apply pretrained model
    data_dict = get_data_sbm(data_dp)
    nr_files = len(data_dict)
    print(f"\t\t> got data from {nr_files} files\n"
          f"\t* Getting saved model")
    model = SequenceTagger.load(model_fp)
    print(f"\t\t> got model\n"
          f"\t* Using pretrained model to extract terms from unseen corpus\n")

    # collect all unique candidate terms in dict, with terms as keys and list of filenames in which they
    # occur as values (include duplicates, so length of list = frequency of term)
    candidate_term_dict = {}

    # collect tokenised string of all data, with "*_*" between sentences and files
    # to easily count total frequency of all terms
    tokenised_texts = ""

    # iterate over files in data_dict
    file_nr = 0
    for output_fn, sentences in data_dict.items():
        fn = output_fn.replace("seqOutput_", "")
        fn = fn.replace(".tsv", ".txt")
        file_nr += 1
        print(f"\t\t> working on file {file_nr}/{nr_files}:\t{output_fn}")
        output_fp = output_dp + output_fn
        pred_sentences = []                         # triple-nested list: sentences, tokens, labels

        # iterate over sentences per file
        for sentence_list in sentences:
            sentence = Sentence(sentence_list)      # [token1, token2, ...]
            tokenised_texts += " ".join(sentence_list) + "*_*"

            # use pretrained model to tag sentence
            model.predict(sentence)
            sentence_pred_string = sentence.to_tagged_string()              # "token1 token2 <I> token3 token4 <B>"
            split_sentence_pred_string = sentence_pred_string.split()   # [token1, token2, <I>, token3, token4, <B>]

            # turn prediction in string form into nested list with [token, predicted_label] pairs
            # and include the "O" label (so "I", "O", or (if applicable) "B")
            sentence_pred_list = []
            last_term = ""
            for index, i in enumerate(split_sentence_pred_string):
                if iob_or_io.lower() == "iob":
                    if i not in ["<I>", "<B>"]:
                        if index == (len(split_sentence_pred_string) - 1) \
                                or split_sentence_pred_string[index + 1] not in ["<I>", "<B>"]:
                            sentence_pred_list.append([i, "O"])
                            if last_term:
                                if last_term.strip() not in candidate_term_dict:
                                    candidate_term_dict[last_term.strip()] = [fn]
                                    last_term = ""
                                else:
                                    candidate_term_dict[last_term.strip()].append(fn)
                                    last_term = ""
                        else:
                            sentence_pred_list.append([i, split_sentence_pred_string[index + 1][1]])
                            if split_sentence_pred_string[index + 1] == "<I>":
                                last_term += i + " "
                            else:
                                if last_term:
                                    if last_term not in candidate_term_dict:
                                        candidate_term_dict[last_term.strip()] = [fn]
                                        last_term = i + " "
                                    else:
                                        candidate_term_dict[last_term.strip()].append(fn)
                                        last_term = i + " "
                                else:
                                    last_term = i + " "
                    elif index == (len(split_sentence_pred_string) - 1):
                        if last_term:
                            if last_term.strip() not in candidate_term_dict:
                                candidate_term_dict[last_term.strip()] = [fn]
                                last_term = ""
                            else:
                                candidate_term_dict[last_term.strip()].append(fn)
                                last_term = ""
                else:   # IO
                    if i != "<I>":
                        if index == (len(split_sentence_pred_string) - 1) \
                                or split_sentence_pred_string[index + 1] != "<I>":
                            sentence_pred_list.append([i, "O"])
                            if last_term:
                                if last_term.strip() not in candidate_term_dict:
                                    candidate_term_dict[last_term.strip()] = [fn]
                                    last_term = ""
                                else:
                                    candidate_term_dict[last_term.strip()].append(fn)
                                    last_term = ""
                        else:
                            sentence_pred_list.append([i, split_sentence_pred_string[index + 1][1]])
                            last_term += i + " "
                    elif index == (len(split_sentence_pred_string) - 1):
                        if last_term:
                            if last_term not in candidate_term_dict:
                                candidate_term_dict[last_term.strip()] = [fn]
                                last_term = ""
                            else:
                                candidate_term_dict[last_term.strip()].append(fn)
                                last_term = ""
            pred_sentences.append(sentence_pred_list)

        with open(output_fp, "wt", encoding="utf-8") as output_f:
            for sentence in pred_sentences:
                for token_pred in sentence:
                    output_f.write("\t".join(token_pred) + "\n")
                output_f.write("\n")

    # write termlist of combined corpus to separate file
    termlist_fp = output_dp + "combined_termlist.tsv"
    capscorrected_candidate_term_dict = map_capitalisation_termlist(candidate_term_dict)
    sorted_term_list = list(sorted(capscorrected_candidate_term_dict.items(), key=lambda x: len(x[1]), reverse=True))
    with open(termlist_fp, "wt", encoding="utf-8") as termlist_f:
        for candidate_term, file_list in sorted_term_list:
            file_list.sort()
            sorted_file_list = list(set(file_list))
            sorted_file_list_string = ", ".join(sorted_file_list)
            ct_lower_with_spaces = " " + candidate_term.lower() + " "
            ct_lower_begin = "*_*" + candidate_term.lower() + " "
            ct_lower_end = " " + candidate_term.lower() + "*_*"
            ct_single = "*_*" + candidate_term.lower() + "*_*"
            total_freq = tokenised_texts.lower().count(ct_lower_with_spaces)
            total_freq += tokenised_texts.lower().count(ct_lower_begin)
            total_freq += tokenised_texts.lower().count(ct_lower_end)
            total_freq += tokenised_texts.lower().count(ct_single)
            if tokenised_texts.startswith(candidate_term.lower() + " "):
                total_freq += 1
            if tokenised_texts.endswith(" " + candidate_term.lower()):
                total_freq += 1
            to_write = f"{candidate_term}\t{total_freq}\t{len(file_list)}\t{sorted_file_list_string}\n"
            termlist_f.write(to_write)

    print(f"\n3. COMPLETED sequential term extraction using multilingual BERT;\n"
          f"\t* Wrote sequential results per file to separate files in {output_dp}\n"
          f"\t* Wrote list of unique candidate terms for entire corpus to {termlist_fp}\n"
          f"\t\t> found {len(list(capscorrected_candidate_term_dict.keys()))} unique candidate terms\n")
    return output_dp


prep_corpus_sbm("unseen_corpora/mono_test/", "en", tok_nesting="eos")
extract_terms_sbm("unseen_corpora/mono_test/", ["corp", "equi", "htfl", "wind"], "iob", optimiser="AdamW",
                  nr_hidden=1, size=512, incl_incorr_tok=True, specific=1, common=1, ood=1, ne=1, partial=1)
