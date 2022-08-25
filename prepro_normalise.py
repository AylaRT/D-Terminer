"""script for normalisation
WARNING: exact copy of part of prepro_normalise.py in ID_TERM repo!
"""

import unicodedata


def normalise_dashes_quotes(text):
    """
    Given a string, replace all dashes and quotes (both single and double quotes)
    of various types of encoding, to be the same.

    :param text = string to normalise
    :return string_normalised
    """
    dashes = ["-", "−", "‐"]
    double_quotes = ['"', '“', '”', '„', "„", "„"]
    single_quotes = ["'", "`", "´", "’", "‘", "’"]

    for double_quote in [',,', "''", "''", "‘’", "’’"]:
        if double_quote in text:
            text = text.replace(double_quote, '"')

    string_normalised = ""
    for char in text:
        if char in dashes:
            string_normalised += "-"
        elif char in double_quotes:
            string_normalised += '"'
        elif char in single_quotes:
            string_normalised += "'"
        else:
            string_normalised += char

    return string_normalised


def normalise_accented_i(text):
    """
    Given a string (of any length), replace all uppercase i's with a dotted accent ("İ")
    to a normal uppercase "I", since this accent causes trouble when lowercasing the data,
    since the lowercase i already has the dot above it.

    :param text: input string
    :return: normalised_text where any "İ" has been replaced by "I"
    """
    normalised_text = text.replace("İ", "I")
    return normalised_text


def normalise_unicode(text):
    """
    Given an input string, normalise this text to use the same kind of unicode
    (avoids encoding problems), using the unidecode package.

    :param text: input string
    :return: normalised_text: same string but with unidecode applied to it
    """
    normalised_text = unicodedata.normalize("NFC", text)
    return normalised_text


def normalise_text_standard(text, unidecoded=True, dashes_quotes=True, accented_i=True, lowercase=False):
    """
    Given an input string, apply the standard normalisation procedure
    in the correct order (with customisable steps):
        1. unicodedata.normalise("NFC", content) (normalise_unicode)
        2. normalise_dashes_quotes(content) (normalise_dashes_quotes)
        3. content.replace("İ", "I") (normalise_accented_i)
        4. content.lower()

    :param text: input string
    :param unidecoded: whether to use normalise_unicode
    :param dashes_quotes: whether to normalise dashes and quotes
    :param accented_i: whether to normalise dot-accented, uppercase I's
    :param lowercase: whether to lowercase the text
    :return: normalised text
    """
    if unidecoded:
        text = normalise_unicode(text)
    if dashes_quotes:
        text = normalise_dashes_quotes(text)
    if accented_i:
        text = text.replace("İ", "I")
    if lowercase:
        text = text.lower()
    return text


def normalise_file_standard(in_fp, out_fp, unidecoded=True, dashes_quotes=True, accented_i=True, lowercase=False):
    """
    Apply the standard normalisation procedure to the text in a given filepath
    and write the normalised output to the provided output filepath.
    Order of operations (first check which operations are required).
    See function: normalise_text_standard

    :param in_fp: input filepath with original data
    :param unidecoded: whether to use normalise_unicode
    :param dashes_quotes: whether to normalise dashes and quotes
    :param accented_i: whether to normalise dot-accented, uppercase I's
    :param lowercase: whether to lowercase the text
    :param out_fp: output filepath for normalised data
    """
    with open(in_fp, "rt", encoding="utf-8") as in_f:
        content = in_f.read()
        normalised_content = normalise_text_standard(content, unidecoded=unidecoded, dashes_quotes=dashes_quotes,
                                                     accented_i=accented_i, lowercase=lowercase)
        with open(out_fp, "wt", encoding="utf-8") as out_f:
            out_f.write(normalised_content)
