# D-Terminer
Repository for the LT3 D-Terminer demo on monolingual and multilingual automatic term extraction, based on the PhD of Ayla Rigouts Terryn (D-TERMINE: Data-driven Term Extraction Methodologies Investigated).

Last updated: 01/08/2022

Contact: ayla.rigoutsterryn@kuleuven.be


## References for Academic Use

Please refer to the following paper for academic use of the demo:

Rigouts Terryn, A., Hoste, V., & Lefever, E. (2022). Tagging Terms in Text: A Supervised Sequential Labelling Approach to Automatic Term Extraction. Terminology. International Journal of Theoretical and Applied Issues in Specialized Communication, 28(1). https://doi.org/10.1075/term.21010.rig

This paper describes the dataset and sequential method for monolingual automatic term extraction in detail. There is no paper yet on the multilingual methodology, but the reference will be added once a paper is published. 


## Document Structure:
1. Project Directories

   1.1 LeTs

   1.2 pretrained_models

   1.3 unseen_corpora

2. ACTER dataset (training and test data)

   2.1 Languages and Domains

   2.2 Term Labels 

3. Monolingual Automatic Term Extraction

   3.1 Description   

   3.2 Involved code files

   3.3 Methodologies
        
      * sequential Bert-multi ("sbm")
         * description
         * use
         * configurable Settings

4. Multilingual Automatic Term Extraction

   4.1 Description

   4.2 Use


## 1. Project Directories

### 1.1 LeTs

Preprocessing (currently only relevant for tokenisation) is performed with LeTs Preprocess. This package is currently excluded from the repo, but can be consulted at https://github.ugent.be/lt3/lets 


### 1.2 pretrained_models

Under the "pretrained_models" directory, you will find different subdirectories per methodology, with informative names. Currently, the only one available is "seq_bert_multi", but this will be elaborated.


### 1.3 unseen_corpora

Unseen corpora (the input for the term extraction) should be stored under this directory, with a separate subdirectory per corpus. For each project subdirectory, the corpus itself (the .txt or .tmx files) should be stored under the subdirectory "corpus". Additional subdirectories (for preprocessed data and results) will be created automatically.


## 2. ACTER dataset

### 2.1 Description

The ACTER dataset is described in detail in the following publications:

**main publication:**

Rigouts Terryn, A., Hoste, V., & Lefever, E. (2020). In No Uncertain Terms: A Dataset for Monolingual and Multilingual Automatic Term Extraction from Comparable Corpora. Language Resources and Evaluation, 54(2), 385–418. https://doi.org/10.1007/s10579-019-09453-9

**other relevant publications:**

Rigouts Terryn, A., Hoste, V., & Lefever, E. (2018). A Gold Standard for Multilingual Automatic Term Extraction from Comparable Corpora: Term Structure and Translation Equivalents. Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC), 1803–1808.

Rigouts Terryn, A. (2021). D-TERMINE: Data-driven Term Extraction Methodologies Investigated [Doctoral thesis]. Ghent University.

Rigouts Terryn, A., Hoste, V., Drouin, P., & Lefever, E. (2020). TermEval 2020: Shared Task on Automatic Term Extraction Using the Annotated Corpora for Term Extraction Research (ACTER) Dataset. Proceedings of the 6th International Workshop on Computational Terminology (COMPUTERM 2020), 85–94.


**Links**:

ACTER on Github: https://github.com/AylaRT/ACTER

ACTER on CLARIN: http://hdl.handle.net/20.500.12124/38


**details**:

The ACTER "Annotated Corpora for Term Extraction Research" dataset consists of corpora in three languages in four domains, in which terms have been manually annotated. It is freely available online (on Github and CLARIN) and was used for the TermEval20202 shared task.


### 2.2 Languages and Domains

ACTER contains corpora in three languages (English, French, Dutch) and four domains (corruption, dressage (horse riding), heart failure, wind energy). For each domain, there is a comparable corpus in all three languages, i.e., the corpus in each language contains texts with a similar subject and style and the size of the corpus per language is similar as well. Only the corpora on corruption contain direct translations. In summary, there are three comparable trilingual corpora and there is one parallel trilingual corpus.

* **corruption**: mostly legal texts by EU institutions, Transparency International, and NATO; many Named Entities and fewer Specific Terms.

* **dressage (horse riding)**: texts from online magazines and blogs, so more informal; many terms (both Common and Specific) and many non-nominal terms (verbs and adjectives) as well.

* **heart failure**: medical abstracts and short papers with many terms, including many Specific Terms.

* **wind energy**: mostly technical documents and papers; not as many terms as in dressage or heart failure, but more Specific Terms than corruption. 


### Term Labels

The manual annotations in the ACTER corpus consist of 4 labels:
* Specific Terms (lexicon-specific and domain-specific: terms in the strictest sense of the word)
* Common Terms (domain-specific, not lexicon-specific: relevant to the domain but known to laypeople)
* Out-of-Domain Terms (lexicon-specific, not domain-specific: not relevant to domain, but not generally known)
* Named Entities (proper names of people, places, organisations, etc.)

For more information, see: https://doi.org/10.1007/s10579-019-09453-9



## 3. Monolingual Automatic Term Extraction


### 3.1 Description

The monolingual term extraction in the D-Terminer demo is based on the research in the D-TERMINE PhD of Ayla Rigouts Terryn (http://hdl.handle.net/1854/LU-8709150). The current version of the demo only covers the neural sequential methodology, but will be elaborated.

Generally, many improvements are planned over time for the entire monolingual pipeline, including more advanced linguistic preprocessing and customisable exports of the results.


### 3.2 Involved Code Files

Main code: seq_bert_multi.py

Preprocessing: prepro_normalise.py and prepro_linguistic.py

Other: dterminer_reusables.py


### 3.3 Methodologies

#### 3.3.1 sequential bert-multi ("seq_bert_multi", "sbm") 

##### description

Sequential term extraction (per token, with IO(B) scheme) using the Flair framework to implement a recurrent neural network using multilingual BERT embeddings. Based on the work presented in https://doi.org/10.1075/term.21010.rig and using the ACTER dataset to pretrain the models.


#### use

**input**: Corpus of plain text (".txt") files; can be multiple files per corpus. Create a subdirectory under "unseen_corpora" with the name of the corpus and store the text files under a subdirectory "corpus" (D-Terminer/unseen_corpora/corpus_name/corpus).

**process**: The corpus will be prepared (minor normalisation, tokenisation + sentence-splitting) and written to separate files with one token per line and an empty line between sentences. This prepared corpus will be written to D-Terminer/unseen_corpora/corpus_name/data_seq_no_features (i.e., sequential data without features). A pretrained model will be used to tag each token as part of a term or not, according to the settings (see further).

**result**: The result will be stored under D-Terminer/unseen_corpora/corpus_name/output_seq_bert_multi with a separate directory that is assigned a name based on the experiment settings. It contains as many files as the original corpus. There will be one token per line, followed by a tab and the assigned label (I, O, or B), and with empty lines between sentences. The results will also be presented as a list of unique candidate terms in that same directory with the filename combined_termlist.tsv. This file has one candidate term per line, followed by a tab and the candidate term frequency, followed by another tab and the list of all files in which the candidate term occurs. The candidate terms in this document are sorted based on frequency. Original casing is maintained, meaning that identical candidate terms with different capitalisations are not combined. 


#### configurable settings

* **language**: "en", "fr", "nl", "de" (indicate language of unseen corpus, i.e.,  English, French, Dutch, German) 
  * Language needs to be indicated for tokenisation purposes, and LeTs Preprocess currently handles only these 4 languages. Otherwise, language is not that relevant for these models, in the sense that models will always be trained on all available languages in the training data (English, French, Dutch) and these models can be used for these and other languages. 

* **domain**: ["corp", "equi", "htfl", "wind"] (list all domains to be used for training, i.e., corruption, equitation (dressage), heart failure, wind energy)
  * You can choose which domains should be included in the training data. This is relevant because term characteristics differ substantially per domain, so the more relevant the training data is, the better the results will be. If the unseen corpus does not resemble any of the domains in the training data, it is probably best to include them all. 

* **term label (specific, common, ood, ne)**: 0, 1 (per parameter, 0 to exclude and 1 to include these labels, i.e., Specific Terms, Common Terms, Out-of-Domain terms, Named Entities)
  * You can customise which instances should be extracted as terms, and which should be ignored. The training data contains annotations with 4 labels and the models can be trained to extract instances with all of these labels, or only those with specific labels. The standard setting is to extract instances of all 4 types (all 4 parameters = 1) and this also tends to lead to the best results. However, you can set the parameters of one or more labels to 0. Not all combinations are currently possible because some are more logical than others. 

* **iob or io**: "io", "iob" (indicate which sequential labelling scheme should be used)
  * The sequential labelling of tokens can either be done in a binary way (each token is either inside (I) or outside (O) of a term), or according to the IOB scheme, which has a separate label for the first word in a term. While the binary labelling scheme can obtain higher sequential scores (classifying two labels is easier than choosing between three), it cannot distinguish between two terms that follow each other without a token in between to separate them. That is only possible with the IOB labelling scheme. Both approaches perform roughly equally well in most cases when comparing the lists of extracted terms.


## 4. Multilingual Automatic Term Extraction

### 4.1 Description

The currently available multilingual automatic term extraction relies on parallel corpora (i.e., sentence-aligned translations), provided as .tmx files.

In a first step, the monolingual texts are reconstructed from the .tmx file and saved as separate .txt files. These are used for monolingual automatic term extraction (currently with the sbm pipeline) and treated in exactly the same way as when only monolingual term extraction has to be performed.

After getting the results for the monolingual term extraction per language, word alignment is performed with the ASTrED package (https://lt3.ugent.be/astred-demo/). Using these word alignments, potential cross-lingual equivalents are suggested for each candidate term in the source language, based on the candidate terms in the target language.


### 4.2 Use and Involved Files

**involved scripts**:  

Main code: multilingual.py

Monolingual ATE: seq_bert_multi.py, prepro_normalise.py and prepro_linguistic.py

Other: dterminer_reusables.py

**use**:

You can either run the steps of the multilingual pipeline separately (prep_multilingual_ate, mono_sbm_ate_for_multilingual, tok_corpus_dp_to_dicts, multilingual_ate_sbm) or at once with a function that combines them all (multilingual_ate_sbm_complete).

As with the monolingual pipeline: create a project directory under "unseen_corpora". Save the corpus as one or more .tmx files under the subdirectory "corpus". A separate project directory under "unseen_corpora" will be created per language, which will contain the monolingual corpora and results of the monolingual term extraction. The results (alignments) of the multilingual term extraction will be saved in the subdirectory "output_multilingual" under the original project directory.

Existing files will not be overwritten without warning.
