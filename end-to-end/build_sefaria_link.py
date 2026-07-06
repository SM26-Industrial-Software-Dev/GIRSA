import sqlite3

#turns the dict with two addresies into a sefaria link
def create_link(dict):
    book = dict.get("book")
    addr1 = dict.get("addr1")
    addr2 = dict.get("addr2")
    path = (str(addr1) , str(addr2) , "1", "1")
    conn, curr = connect()
    depth = find_depth(curr, book)
    conn.close()

    book = book.replace(" ", "_")
    link = f"https://www.sefaria.org/{book}"
    for addr in range(depth):
        link = link + "." + path[addr]
    return link
        
def connect():
    conn = sqlite3.connect('sefaria.db')
    curr = conn.cursor()
    return conn, curr

def find_depth(curr, book):
    lookup = (f"""SELECT depth FROM books WHERE id = "{book}";""")
    curr.execute(lookup)
    return curr.fetchone()[0]