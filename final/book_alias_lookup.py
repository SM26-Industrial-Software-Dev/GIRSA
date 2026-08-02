import sqlite3
from rapidfuzz import fuzz

# returns an ordered list of all probable matches for this source in order
def find_in_aliases(name, similarity_percentage = 85):
    sp = __normalize(similarity_percentage)
    book_info = __get_names()
    results = __calc_similarity(book_info, name, sp)
    return list(results.keys())

# returns True if there is a probable match for this source
def is_probably_in_aliases_(name, similarity_percentage = 85):
    return bool(find_in_aliases(name, similarity_percentage))

#  one location e.g. yerushalmi on shabbos, rayved on the rambam
#  two locations e.g. pnei moshe on yerushalmi on shabbos, rayved on the rambam hilchos yesodei hatorah
#  returns an ordered list of all probable matches for this source in order
#  order is currently determind based on a weighted average of the similarity of the name and the similarity of the location(s)
#  wheights can be adjusted below (lines 36-40), the current values are based on a few test cases and may not be optimal for all cases
def find_in_aliases_with_location(name, location1, location2 = None, similarity_percentage_name = 70, total_similarity_percentage = 85):
        spn = __normalize(similarity_percentage_name)
        spt = __normalize(total_similarity_percentage)

        book_info = __get_names()
        results1 = __calc_similarity(book_info, name, spn)

        location1_info = __get_locations(list(results1.keys()))
        results2 = __calc_similarity(location1_info, location1, 0)

        results3 = {}
        if location2 is not None:
            location2_info = __get_locations(list(results1.keys()))
            results3 = __calc_similarity(location2_info, location2, 0)
        else:
            results3 = {}

        results = {}
        for book_name in results1:
            if book_name in results2 and book_name not in results3:
                results[book_name] = results1[book_name]*.65 + results2[book_name]*.35
            elif book_name not in results2 and book_name in results3:
                results[book_name] = results1[book_name]*.65 + results3[book_name]*.35
            else: 
                results[book_name] = results1[book_name]*.60 + results2[book_name]*.20 + results3[book_name]*.20
        
        results =dict(sorted(filter(lambda x: x[1] >= spt, results.items()), key=lambda x: x[1], reverse=True))
        return list(results.keys())
    

def __calc_similarity(corpus, name, sp):
    results = {}
    for book_name in corpus:
        score = fuzz.partial_ratio(name.lower(), book_name[1].lower().replace("-", " "))
        if score >= sp:
            results[book_name[0]] = max(__previous(results, book_name), score)
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return results
    
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

if __name__ == "__main__":
    print(find_in_aliases("penei moshe"))