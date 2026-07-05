import sqlite3

def create_link(book, addr1 = 0, addr2 = 0, addr3 = 0, addr4 = 0):
    path = (str(addr1 + 1) , str(addr2 + 1) , str(addr3 + 1) , str(addr4 + 1))
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