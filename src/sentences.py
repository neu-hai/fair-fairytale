from nltk.tokenize import sent_tokenize

def split_sentences(story_txt: str):
    '''Split sentences by period and semi-colon, accounting for dialogue.'''
    sentences = sent_tokenize(story_txt)
    new_sentences = []

    for sentence in sentences:
        if ';' in sentence:
            clauses = sentence.split(';')
            new_clauses = []
            for clause in clauses:
                # Do we take out the space after the semi-colon?
                if clause[-1].isalpha():
                    new_clause = clause + ';' 
                    new_clauses.append(new_clause)
                else:
                    new_clause = clause
                    new_clauses.append(new_clause)
            new_sentences += new_clauses
        else:
            new_sentences.append(sentence)
            
    return new_sentences