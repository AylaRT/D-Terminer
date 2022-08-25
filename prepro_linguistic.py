"""Use LeTs for linguistic preprocessing"""


from lets.tokenizer import Tokenizer


def tokenise_text(text, language, nesting="eos"):
    """
    Use LeTs to tokenise a text, given as a string, in the specified language.

    :param text: text to be preprocessed as string
    :param language: "en", "fr", "nl", or "de"
    :param nesting: "eos" => nested for each sentence
                    "newlines" => only nesting for newlines in original text
                    "none" => no nesting
    :return: nested list of tokenised sentences
    """
    if nesting not in ["eos", "newlines", "none"]:
        raise ValueError(f"ERROR tokenise_text: 'nesting' parameter should be 'eos', 'newlines', or 'none'; "
                         f"not {nesting}")
    tokeniser = Tokenizer(l=language)
    tokenised = []
    for line in text.splitlines():
        tokenised_line = tokeniser.process_line(line)
        tok_sentence = []
        tok_line = []
        if tokenised_line:
            for token in tokenised_line:
                if token:
                    if nesting == "none":
                        tokenised.append(token)
                    elif nesting == "newlines":
                        tok_line.append(token)
                    elif nesting == "eos":
                        tok_sentence.append(token)
                elif tok_sentence:
                    if nesting == "eos":
                        if tok_sentence:
                            tokenised.append(tok_sentence)
                            tok_sentence = []
            if tok_line:
                tokenised.append(tok_line)
    return tokenised
