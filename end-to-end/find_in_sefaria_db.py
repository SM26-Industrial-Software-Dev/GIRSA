import sqlite3
from rapidfuzz import fuzz

# returns an ordered list of all probable matches for this source in order
def find_in_aliases(name, similarity_percentage = 85):
    sp = __normalize(similarity_percentage)
    book_info = __get_names()
    results = __calc_similarity(book_info, name, sp)
    return results

def is_probably_in_aliases_(name, similarity_percentage = 85):
    return bool(find_in_aliases(name, similarity_percentage))

#  one location e.g. yerushalmi on brachos, rayved on the rambam
#  two locations e.g. pnei moshe on yerushalmi on brachos, rayved on the rambam hilchos yesodei hatorah
#  order is currently determind based on the last imput - this will be fixed in the future
def find_in_aliases_with_location(name, location1, location2 = None, similarity_percentage_name = 75, similarity_percentage_location1 = 75, similarity_percentage_location2 = 75):
        spn = __normalize(similarity_percentage_name)
        spl1 = __normalize(similarity_percentage_location1)
        spl2 = __normalize(similarity_percentage_location2)
        
        if location2 is not None:
            location_info = __get_locations(with_location)
            with_location =  __calc_similarity(location_info, location2, spl2)
        with_name = find_in_aliases(name, spn)
        location_info = __get_locations(with_name)
        with_location = __calc_similarity(location_info, location1, spl1)

        return with_location
    

def __calc_similarity(corpus, name, sp):
    results = {}
    for book_name in corpus:
        score = fuzz.partial_ratio(name.lower(), book_name[1].lower().replace("-", " "))
        if score >= sp:
            results[book_name[0]] = max(__previous(results, book_name), score)
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return list(results.keys())
    
def __get_names():
    conn = sqlite3.connect('sefaria.db')
    curr = conn.cursor()
    books_info = ('SELECT book_id, alias FROM book_aliases;')
    curr.execute(books_info)
    results = curr.fetchall()
    conn.close()
    return results

def __get_locations(locations):
    conn = sqlite3.connect('sefaria.db')
    curr = conn.cursor()
    results = []
    for location in locations:
        location_info = (f'SELECT book_id, alias FROM book_aliases WHERE book_id = "{location}";')
        curr.execute(location_info)
        results = results + curr.fetchall()
    conn.close()
    return results


def __normalize(num):
    if num > 100:
        num = 100
    elif num < 0:
        num = 0
    return num

def __previous(results, book_name):
    try:
        return results[book_name[0]]
    except (KeyError):
        return 0.0