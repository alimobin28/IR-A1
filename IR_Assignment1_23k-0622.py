'''
'''
import os, re, tkinter as tk
from tkinter import ttk, font as tkfont
from nltk.stem import PorterStemmer

STOPWORDS = {
    "a","is","the","of","all","and","to","can","be","as","once","for",
    "at","am","are","has","have","had","up","his","her","in","on","no",
    "we","do"
}

_ps    = PorterStemmer()
_PUNCT = re.compile(r"[^a-zA-Z0-9]")

'''
This function reads stopwords from a file.
It also adds predefined stopwords.
Finally, it returns a complete set of stopwords.
'''
def load_stopwords():
    sw = set(STOPWORDS)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "Stopword-List.txt"),
                  "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                sw.update(line.split())
    except FileNotFoundError:
        pass
    return sw


STOP_WORDS = load_stopwords()

'''
This function cleans a word by removing punctuation.
It converts the word into lowercase.
It returns a valid cleaned token.
'''
def clean_token(w: str) -> str:
    w2 = _PUNCT.sub("", w.lower())
    return "" if (not w2 or w2.isdigit()) else w2

'''
This function preprocesses a word.
It removes stopwords and applies stemming.
It returns the final processed term.
'''
def preprocess(w: str) -> str:
    c = clean_token(w)
    if not c or c in STOPWORDS:
        return ""
    return _ps.stem(c)

'''
This function splits text into tokens.
It uses spaces and punctuation as separators.
It returns a list of words.
'''
def tokenize_doc(text: str) -> list:
    tokens, cur = [], ""
    for ch in text:
        if ch in ' .\n-—?"…/':
            if cur:
                tokens.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        tokens.append(cur)
    return tokens

'''
This function splits query into tokens.
It converts AND, OR, NOT into uppercase.
It helps in Boolean query processing.
'''
def tokenize_query(query: str) -> list:
    query = query.replace("(", " ( ").replace(")", " ) ")
    tokens = []
    for t in query.split():
        tokens.append(t.upper() if t.lower() in ("and", "or", "not") else t)
    return tokens

'''
This function extracts document ID from filename.
It removes prefix and extension.
It returns numeric ID.
'''
def get_doc_id(filename: str) -> str:
    return filename.replace("speech_", "").replace(".txt", "")

'''
This function builds inverted and positional indexes.
It reads all documents from folder.
It returns both indexes for searching.
'''
def build_indexes():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory  = os.path.join(script_dir, "Trump Speechs")

    inverted   = {}
    positional = {}

    for filename in sorted(os.listdir(directory)):
        if not (filename.startswith("speech_") and filename.endswith(".txt")):
            continue

        with open(os.path.join(directory, filename),
                  "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        if len(lines) < 2:
            continue

        doc_id = get_doc_id(filename)
        pos    = -1

        for raw_tok in tokenize_doc(lines[1]):
            pos += 1

            c = clean_token(raw_tok)
            if not c:
                continue
            if c in STOPWORDS:
                continue

            term = _ps.stem(c)
            inverted.setdefault(term, set()).add(doc_id)
            positional.setdefault(term, {}).setdefault(doc_id, []).append(pos)

    return inverted, positional

'''
This function creates a universal set of documents.
It collects all document IDs.
It is used in NOT queries.
'''
def universal_set(inverted: dict) -> set:
    u = set()
    for ids in inverted.values():
        u |= ids
    return u

'''
This function returns posting list of a term.
It searches term in inverted index.
It returns a set of documents.
'''
def get_posting(term: str, inverted: dict) -> set:
    return set(inverted.get(term, set()))

'''
This function processes a query word.
It handles special spelling cases.
It returns matching documents.
'''
def resolve_term(word: str, inverted: dict) -> set:
    t = preprocess(word)
    return get_posting(t, inverted) if t else set()

'''
This function handles single word query.
It calls resolve_term.
It returns result set.
'''
def single_term_query(word: str, inverted: dict) -> set:
    return resolve_term(word, inverted)

'''
This function handles phrase queries.
It checks if two words appear together.
It uses positional index.
'''
def phrasal_query(words: list, inverted: dict, positional: dict) -> set:
    if len(words) != 2:
        return set()
    c1 = clean_token(words[0])
    c2 = clean_token(words[1])
    if not c1 or not c2:
        return set()
    if c1 in STOPWORDS or c2 in STOPWORDS:
        return set()
    t1 = _ps.stem(c1)
    t2 = _ps.stem(c2)

    common = get_posting(t1, inverted) & get_posting(t2, inverted)
    result = set()
    for doc_id in common:
        p1_list = positional.get(t1, {}).get(doc_id, [])
        p2_set  = set(positional.get(t2, {}).get(doc_id, []))
        if any((p + 1) in p2_set for p in p1_list):
            result.add(doc_id)
    return result

'''
This function handles proximity queries.
It finds words within distance k.
It returns matching documents.
'''
def proximity_query(words: list, inverted: dict, positional: dict) -> set:
    k = int(words[-1].lstrip("/"))
    w1, w2 = words[0], words[1]
    t1 = preprocess(w1)
    t2 = preprocess(w2)
    if not t1 or not t2:
        return set()

    common = get_posting(t1, inverted) & get_posting(t2, inverted)
    result = set()
    for doc_id in common:
        p1_list = positional.get(t1, {}).get(doc_id, [])
        p2_set  = set(positional.get(t2, {}).get(doc_id, []))
        if any((p + k + 1) in p2_set for p in p1_list):
            result.add(doc_id)
    return result

'''
This function peeks at the current token without consuming it.
It returns None if there are no more tokens left.
'''
def peek(tokens, idx):
    if idx[0] < len(tokens):
        return tokens[idx[0]]
    return None

'''
This function consumes and returns the current token.
It moves the index forward by one position.
'''
def consume(tokens, idx):
    token = tokens[idx[0]]
    idx[0] += 1
    return token

'''
This function parses a Boolean expression with AND and OR operators.
It keeps combining terms as long as AND or OR appears.
It returns the final result set.
'''
def parse_expr(tokens, idx, inverted, universe):
    left = parse_term(tokens, idx, inverted, universe)
    while peek(tokens, idx) in ("AND", "OR"):
        op    = consume(tokens, idx)
        right = parse_term(tokens, idx, inverted, universe)
        if op == "AND":
            left = left & right
        else:
            left = left | right
    return left

'''
This function parses a single term which can be NOT, brackets, or a word.
It handles all three cases and returns a result set.
'''
def parse_term(tokens, idx, inverted, universe):
    tok = peek(tokens, idx)
    if tok == "NOT":
        consume(tokens, idx)
        return universe - parse_term(tokens, idx, inverted, universe)
    elif tok == "(":
        consume(tokens, idx)
        result = parse_expr(tokens, idx, inverted, universe)
        if peek(tokens, idx) == ")":
            consume(tokens, idx)
        return result
    else:
        return resolve_term(consume(tokens, idx), inverted)

'''
This function starts Boolean query processing.
It uses a mutable index list to track position across recursive calls.
It returns final result.
'''
def boolean_query(tokens, inverted, positional, universe) -> set:
    idx = [0]
    return parse_expr(tokens, idx, inverted, universe)


OPERATORS = {"AND", "OR", "NOT"}

'''
This function decides query type.
It calls appropriate handler.
It returns final result.
'''
def process_query(raw_query: str, inverted: dict, positional: dict, universe: set) -> set:
    tokens = tokenize_query(raw_query.strip())
    if not tokens:
        return set()

    if any(re.match(r"^/\d+$", t) for t in tokens):
        return proximity_query(tokens, inverted, positional)

    if len(tokens) == 1 and tokens[0] not in OPERATORS:
        return single_term_query(tokens[0], inverted)

    if any(t in OPERATORS or t in ("(", ")") for t in tokens):
        return boolean_query(tokens, inverted, positional, universe)

    if len(tokens) == 2:
        return phrasal_query(tokens, inverted, positional)

    result = set(universe)
    for tok in tokens:
        result &= resolve_term(tok, inverted)
    return result

'''
This function formats result output.
It sorts document IDs.
It returns readable string.
'''
def format_result(result: set) -> str:
    if not result:
        return "No results found."
    return "{" + ", ".join(f"'{d}'" for d in sorted(result, key=lambda x: int(x))) + "}"

'''
This function builds and launches the graphical user interface.
It creates input box, buttons, and result display area.
It allows user to enter queries and see results.
'''
def launch_gui(inverted, positional, universe):
    root = tk.Tk()
    root.title("Boolean Retrieval Model")
    root.geometry("400x400")
    root.resizable(False, False)
    root.configure(bg="#ffffff")

    tf = tkfont.Font(family="Courier", size=11, weight="bold")
    lf = tkfont.Font(family="Courier", size=9)
    rf = tkfont.Font(family="Courier", size=9)

    '''
    This section sets styles for buttons.
    It defines colors, fonts, and hover effects.
    It improves the GUI appearance.
    '''
    st = ttk.Style()
    st.theme_use("clam")
    st.configure("TButton", background="#1a56db", foreground="#ffffff",
                 font=("Courier", 10, "bold"), padding=6, relief="flat")
    st.map("TButton", background=[("active", "#1344b5")])
    st.configure("C.TButton", background="#e8f0fe", foreground="#1a56db",
                 font=("Courier", 10, "bold"), padding=6, relief="flat")
    st.map("C.TButton", background=[("active", "#d0e2ff")])

    '''
    This section creates the title bar.
    It displays the heading of the application.
    '''
    title_bar = tk.Frame(root, bg="#1a56db")
    title_bar.pack(fill="x")
    tk.Label(title_bar, text="BOOLEAN RETRIEVAL MODEL",
             bg="#1a56db", fg="#ffffff", font=tf).pack(pady=12)

    tk.Frame(root, height=2, bg="#1a56db").pack(fill="x")

    '''
    This section creates input field for query.
    User enters search query here.
    '''
    tk.Label(root, text="Enter Query:", bg="#ffffff",
             fg="#1a56db", font=lf).pack(anchor="w", padx=24, pady=(14, 2))

    qv = tk.StringVar()
    ent = tk.Entry(root, textvariable=qv, width=44,
                   bg="#e8f0fe", fg="#1a1a2e", insertbackground="#1a56db",
                   font=("Courier", 10), relief="solid", bd=1)
    ent.pack(padx=20, pady=(0, 4))
    ent.focus()

    '''
    This label shows error messages.
    It displays warning if query is empty.
    '''
    err = tk.Label(root, text="", bg="#ffffff", fg="#cc0000", font=lf)
    err.pack(anchor="w", padx=24)

    '''
    This section displays the result area.
    It shows retrieved documents.
    '''
    tk.Label(root, text="Result:", bg="#ffffff",
             fg="#1a56db", font=lf).pack(anchor="w", padx=24, pady=(8, 2))

    box = tk.Frame(root, bg="#e8f0fe", bd=1, relief="solid")
    box.pack(padx=20, fill="both", expand=True, pady=(0, 4))

    rt = tk.Text(box, height=8, wrap="word", bg="#e8f0fe", fg="#1a1a2e",
                 font=rf, relief="flat", state="disabled", bd=0,
                 selectbackground="#1a56db", selectforeground="#ffffff")
    sb = tk.Scrollbar(box, command=rt.yview)
    rt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    rt.pack(side="left", fill="both", expand=True, padx=6, pady=6)

    '''
    This function runs when submit button is clicked.
    It processes the query and shows results.
    It also checks for empty input.
    '''
    def submit():
        q = qv.get().strip()
        err.config(text="")
        if not q:
            err.config(text="Please enter a query.")
            return
        res = process_query(q, inverted, positional, universe)
        rt.config(state="normal")
        rt.delete("1.0", "end")
        rt.insert("end", format_result(res))
        rt.config(state="disabled")

    '''
    This function clears input and result.
    It resets the GUI for new query.
    It also focuses back on input field.
    '''
    def clear():
        qv.set("")
        err.config(text="")
        rt.config(state="normal")
        rt.delete("1.0", "end")
        rt.config(state="disabled")
        ent.focus()

    '''
    This section creates buttons.
    Submit button runs query.
    Clear button resets the fields.
    '''
    bf = tk.Frame(root, bg="#ffffff")
    bf.pack(pady=(0, 16))
    ttk.Button(bf, text="  Submit  ", command=submit).pack(side="left", padx=8)
    ttk.Button(bf, text="  Clear  ", style="C.TButton",
               command=clear).pack(side="left", padx=8)

    '''
    This binds Enter key to submit action.
    It allows quick query execution.
    '''
    root.bind("<Return>", lambda e: submit())

    '''
    This starts the GUI loop.
    It keeps the window running.
    '''
    root.mainloop()

'''
This is the main program block.
It builds indexes and starts GUI.
It allows user to interact with system.
'''
if __name__ == "__main__":
    print("Building indexes…")
    inv, pos = build_indexes()
    uni = universal_set(inv)
    print(f"Done — {len(inv)} unique terms, {len(uni)} documents.")
    launch_gui(inv, pos, uni)