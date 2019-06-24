import re


def preprocess_query(query):
    """It is important to preprocess query before use it as train corpus"""
    if not query or type(query) != str:
        return str(query)
    query = query.strip()
    query = query.replace('?', '')  # remove ?
    query = query.lower()  # lower
    query = re.sub(r'([a-z_0-9)])[,] ', r'\1 , ', query)  # sep , in word tail
    query = re.sub(r' ["\']([a-rt-zA-Z_0-9)])', r" '' \1", query)  # sep " in word start, exclude 's
    query = re.sub(r' ["\'](s[a-z0-9_@])', r" '' \1", query)  # sep " in word start, exclude 's
    query = re.sub(r'([a-zA-Z_0-9)])["\'] ', r'\1 `` ', query)  # sep " in word tail
    query = re.sub(r'([a-zA-Z_0-9)])["\']$', r'\1 ``', query)  # sep " in word tail & in string tail
    query = re.sub(r'([a-zA-Z_.)])[’\']s(\s)', r"\1 's\2", query)  # sep 's
    query = re.sub(r'([a-zA-Z_.)])[:](\s)', r"\1 :\2", query)  # sep : in word tail
    query = re.sub(' [ ]+', ' ', query)  # extra whitespace
    return query


def valid(m_str):
    if not any(str(n) in m_str for n in range(0, 9)):
        return False
    return True


def preprocess_sparql(sparql):
    """Preprocess for sparql logic form"""
    sparql = sparql.strip()
    sparql = sparql.lower()  # lower
    sparql = re.sub('\t', ' ', sparql)  # replace \t with ' '
    # 0. remove the meaningless header and \n
    sparql = sparql.replace('#MANUAL SPARQL'.lower(), '')
    sparql = sparql.replace('prefix ns: <http://rdf.freebase.com/ns/>', '')
    sparql = sparql.replace('\n', ' ')  # remove \n
    sparql = re.sub('\n', ' ', sparql)  # remove \n
    # 1. replace entity
    patterns = [possible_entity_pattern for possible_entity_pattern in re.findall('ns:m.[a-z_0-9]*', sparql) if
                valid(possible_entity_pattern)]
    for pattern in patterns:
        sparql = sparql.replace(pattern, '#entity#')
    # 2. remove type header
    sparql = re.sub(r"select distinct[ ]*\?x[ ]*where[ ]*{[ ]*", " ", sparql)  # remove main clause
    sparql = re.sub('}[ ]?$', ' ', sparql)  # remove } in the tail
    sparql = sparql.replace(
        "filter (?x != ?c) filter (!isliteral(?x) or lang(?x) = '' or langmatches(lang(?x), 'en'))",
        '<sparql-header-1> ')  # replace fixed header of one type LF
    sparql = sparql.replace(
        "filter (?x != #entity#) filter (!isliteral(?x) or lang(?x) = '' or langmatches(lang(?x), 'en'))",
        "<sparql-header-2> ")  # replace fixed header of another type LF
    # 3. sep special chars
    sparql = re.sub(r'([a-z_0-9!?) \'#])([{()}])([#, )a-z_0-9!?])', r"\1 \2 \3", sparql)  # sep ( ) in word start
    sparql = re.sub(r'([a-z_0-9!?)\" \'])\^\^([, )a-z_0-9!?])', r"\1 ^^ \2", sparql)  # sep ^^
    sparql = re.sub(r'([a-z_0-9@)])["\'] ', r'\1 `` ', sparql)  # sep " in word tail
    sparql = re.sub(r' ["\']([a-z_0-9@)])', r" '' \1", sparql)  # sep " in word start
    sparql = re.sub(' [ ]+', ' ', sparql).strip()  # extra whitespace
    return sparql


def extract_sketch_from_sparql(sparql):
    predicates = re.findall('ns:[a-z_0-9]*.[a-z_0-9]*.[a-z_0-9]*', sparql)
    sketch = ' # '.join(predicates)
    return sketch


if __name__ == '__main__':
    # The query test preprocess functionality
    test_query = """
    The education institution has a sports team named George Washington Colonials men 's basketball, the basketball's
    team is named "the basektaball" and it says 'i can win this game'  
    is the actor that played in the film mr conservative: goldwater on goldwater
    "This test the end of quotes tail match"
    """
    result = preprocess_query(test_query)
