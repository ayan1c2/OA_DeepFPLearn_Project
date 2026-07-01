from rdflib import Graph

graph = Graph()
graph.parse("ontology/pfas_ontology.ttl", format="turtle")
graph.parse("ontology/pfas_instances.ttl", format="turtle")

query_file = "sparql/02_get_high_risk_compounds.sparql"

with open(query_file, "r") as f:
    query = f.read()

for row in graph.query(query):
    print(row)
