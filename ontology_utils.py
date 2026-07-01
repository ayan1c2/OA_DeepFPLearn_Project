import os
import json
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec


logger = logging.getLogger("OA_DeepFPLearn")
MISSING_ONTOLOGY_NODES = set()

ONTOLOGY_OUTPUT_DIR = "artifacts/ontology"


def build_ontology_graph():
    graph = nx.Graph()

    nodes = {
        "Chemical": "root",
        "PFAS": "chemical_group",
        "Non_PFAS": "chemical_group",

        "PFAS_Sulfonate": "pfas_class",
        "PFAS_Carboxylate": "pfas_class",
        "PFAS_Ether": "pfas_class",
        "Fluorotelomer": "pfas_class",
        "Perfluoroalkyl_Acid": "pfas_class",
        "Polyfluoroalkyl_Substance": "pfas_class",
        "Precursor_PFAS": "pfas_class",
        "Short_Chain_PFAS": "pfas_class",
        "Long_Chain_PFAS": "pfas_class",

        "PFOS": "compound",
        "PFOA": "compound",
        "PFHxS": "compound",
        "PFHxA": "compound",
        "PFNA": "compound",
        "PFDA": "compound",
        "PFBS": "compound",
        "GenX": "compound",
        "ADONA": "compound",
        "FTOH_6_2": "compound",
        "FTOH_8_2": "compound",

        "Liver_Toxicity": "toxicity_endpoint",
        "Developmental_Toxicity": "toxicity_endpoint",
        "Endocrine_Disruption": "toxicity_endpoint",
        "Reproductive_Toxicity": "toxicity_endpoint",
        "Immunotoxicity": "toxicity_endpoint",
        "Neurotoxicity": "toxicity_endpoint",
        "Carcinogenicity": "toxicity_endpoint",
        "Bioaccumulation": "toxicity_endpoint",
        "Persistence": "toxicity_endpoint",

        "Water_Exposure": "exposure_route",
        "Food_Exposure": "exposure_route",
        "Dust_Exposure": "exposure_route",
        "Air_Exposure": "exposure_route",
        "Occupational_Exposure": "exposure_route",

        "Groundwater": "environmental_compartment",
        "Surface_Water": "environmental_compartment",
        "Soil": "environmental_compartment",
        "Sediment": "environmental_compartment",
        "Biota": "environmental_compartment",
        "Wastewater": "environmental_compartment",

        "High_Risk": "risk_level",
        "Medium_Risk": "risk_level",
        "Low_Risk": "risk_level",

        "Regulated": "regulatory_status",
        "Candidate_for_Regulation": "regulatory_status",
        "Emerging_Concern": "regulatory_status",
        "Restricted": "regulatory_status",

        "Monitoring_Required": "decision_action",
        "Further_Testing": "decision_action",
        "Regulatory_Review": "decision_action",
        "Priority_Screening": "decision_action",
        "Risk_Report": "decision_action"
    }

    for node, node_type in nodes.items():
        graph.add_node(node, node_type=node_type)

    edges = [
        ("Chemical", "PFAS", "is_a"),
        ("Chemical", "Non_PFAS", "is_a"),

        ("PFAS", "PFAS_Sulfonate", "has_class"),
        ("PFAS", "PFAS_Carboxylate", "has_class"),
        ("PFAS", "PFAS_Ether", "has_class"),
        ("PFAS", "Fluorotelomer", "has_class"),
        ("PFAS", "Perfluoroalkyl_Acid", "has_class"),
        ("PFAS", "Polyfluoroalkyl_Substance", "has_class"),
        ("PFAS", "Precursor_PFAS", "has_class"),
        ("PFAS", "Short_Chain_PFAS", "has_chain_length"),
        ("PFAS", "Long_Chain_PFAS", "has_chain_length"),

        ("PFOS", "PFAS_Sulfonate", "belongs_to"),
        ("PFHxS", "PFAS_Sulfonate", "belongs_to"),
        ("PFBS", "PFAS_Sulfonate", "belongs_to"),

        ("PFOA", "PFAS_Carboxylate", "belongs_to"),
        ("PFHxA", "PFAS_Carboxylate", "belongs_to"),
        ("PFNA", "PFAS_Carboxylate", "belongs_to"),
        ("PFDA", "PFAS_Carboxylate", "belongs_to"),

        ("GenX", "PFAS_Ether", "belongs_to"),
        ("ADONA", "PFAS_Ether", "belongs_to"),

        ("FTOH_6_2", "Fluorotelomer", "belongs_to"),
        ("FTOH_8_2", "Fluorotelomer", "belongs_to"),

        ("PFOS", "Long_Chain_PFAS", "has_chain_length"),
        ("PFOA", "Long_Chain_PFAS", "has_chain_length"),
        ("PFHxS", "Long_Chain_PFAS", "has_chain_length"),
        ("PFNA", "Long_Chain_PFAS", "has_chain_length"),
        ("PFDA", "Long_Chain_PFAS", "has_chain_length"),

        ("PFBS", "Short_Chain_PFAS", "has_chain_length"),
        ("PFHxA", "Short_Chain_PFAS", "has_chain_length"),
        ("GenX", "Short_Chain_PFAS", "has_chain_length"),
        ("ADONA", "Short_Chain_PFAS", "has_chain_length"),

        ("PFAS_Sulfonate", "Liver_Toxicity", "associated_with"),
        ("PFAS_Sulfonate", "Immunotoxicity", "associated_with"),
        ("PFAS_Sulfonate", "Bioaccumulation", "associated_with"),

        ("PFAS_Carboxylate", "Developmental_Toxicity", "associated_with"),
        ("PFAS_Carboxylate", "Reproductive_Toxicity", "associated_with"),
        ("PFAS_Carboxylate", "Liver_Toxicity", "associated_with"),

        ("PFAS_Ether", "Endocrine_Disruption", "associated_with"),
        ("PFAS_Ether", "Liver_Toxicity", "associated_with"),

        ("Long_Chain_PFAS", "Bioaccumulation", "associated_with"),
        ("Long_Chain_PFAS", "Persistence", "associated_with"),
        ("Short_Chain_PFAS", "Persistence", "associated_with"),

        ("Liver_Toxicity", "High_Risk", "implies"),
        ("Developmental_Toxicity", "High_Risk", "implies"),
        ("Reproductive_Toxicity", "High_Risk", "implies"),
        ("Carcinogenicity", "High_Risk", "implies"),
        ("Immunotoxicity", "Medium_Risk", "implies"),
        ("Endocrine_Disruption", "Medium_Risk", "implies"),
        ("Neurotoxicity", "Medium_Risk", "implies"),
        ("Persistence", "Medium_Risk", "implies"),
        ("Bioaccumulation", "High_Risk", "implies"),

        ("PFOS", "Regulated", "has_status"),
        ("PFOA", "Regulated", "has_status"),
        ("PFHxS", "Restricted", "has_status"),
        ("GenX", "Emerging_Concern", "has_status"),
        ("ADONA", "Emerging_Concern", "has_status"),
        ("PFNA", "Candidate_for_Regulation", "has_status"),

        ("PFAS", "Water_Exposure", "has_exposure_route"),
        ("PFAS", "Food_Exposure", "has_exposure_route"),
        ("PFAS", "Dust_Exposure", "has_exposure_route"),
        ("PFAS", "Air_Exposure", "has_exposure_route"),
        ("PFAS", "Occupational_Exposure", "has_exposure_route"),

        ("PFAS", "Groundwater", "found_in"),
        ("PFAS", "Surface_Water", "found_in"),
        ("PFAS", "Soil", "found_in"),
        ("PFAS", "Sediment", "found_in"),
        ("PFAS", "Biota", "found_in"),
        ("PFAS", "Wastewater", "found_in"),

        ("High_Risk", "Monitoring_Required", "triggers"),
        ("High_Risk", "Regulatory_Review", "triggers"),
        ("High_Risk", "Risk_Report", "triggers"),
        ("Medium_Risk", "Further_Testing", "triggers"),
        ("Medium_Risk", "Priority_Screening", "triggers"),
        ("Emerging_Concern", "Priority_Screening", "triggers"),
        ("Regulated", "Risk_Report", "triggers"),
        ("Restricted", "Regulatory_Review", "triggers")
    ]

    for source, target, relation in edges:
        graph.add_edge(source, target, relation=relation)

    return graph


def train_ontology_embedding(graph, dimensions=32):
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=20,
        num_walks=100,
        workers=1,
        quiet=True
    )

    return node2vec.fit(
        window=8,
        min_count=1,
        batch_words=4
    )


def get_ontology_vector(class_name, embedding_model, dimensions=32):
    normalized_name = str(class_name).strip().replace(" ", "_")

    if normalized_name in embedding_model.wv:
        return embedding_model.wv[normalized_name].astype(np.float32)

    lower_lookup = {str(node).lower(): node for node in embedding_model.wv.index_to_key}
    matched_node = lower_lookup.get(normalized_name.lower())

    if matched_node is not None:
        return embedding_model.wv[matched_node].astype(np.float32)

    if normalized_name not in MISSING_ONTOLOGY_NODES:
        logger.warning("Missing ontology node: %s. Returning zero vector.", normalized_name)
        MISSING_ONTOLOGY_NODES.add(normalized_name)

    return np.zeros((dimensions,), dtype=np.float32)


def export_missing_ontology_nodes(output_dir=ONTOLOGY_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "missing_ontology_nodes.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("missing_node\n")
        for node in sorted(MISSING_ONTOLOGY_NODES):
            f.write(f"{node}\n")
    return path


def get_reasoning_path(graph, source_node, target_node="High_Risk"):
    source_node = str(source_node).strip().replace(" ", "_")

    if source_node not in graph.nodes or target_node not in graph.nodes:
        return []

    try:
        return nx.shortest_path(graph, source=source_node, target=target_node)
    except nx.NetworkXNoPath:
        return []


def infer_decision_actions(graph, node_name):
    node_name = str(node_name).strip().replace(" ", "_")

    if node_name not in graph.nodes:
        return []

    actions = set()

    for target in ["Monitoring_Required", "Further_Testing", "Regulatory_Review", "Priority_Screening", "Risk_Report"]:
        try:
            path = nx.shortest_path(graph, source=node_name, target=target)
            actions.add(target)
        except nx.NetworkXNoPath:
            pass

    return sorted(actions)


def export_ontology(graph, output_dir=ONTOLOGY_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    edge_list_path = os.path.join(output_dir, "pfas_ontology_edges.csv")
    node_list_path = os.path.join(output_dir, "pfas_ontology_nodes.csv")
    graphml_path = os.path.join(output_dir, "pfas_ontology.graphml")
    json_path = os.path.join(output_dir, "pfas_ontology.json")

    node_rows = []
    for node, attrs in graph.nodes(data=True):
        node_rows.append({
            "node": node,
            "node_type": attrs.get("node_type", "unknown")
        })

    edge_rows = []
    for source, target, attrs in graph.edges(data=True):
        edge_rows.append({
            "source": source,
            "target": target,
            "relation": attrs.get("relation", "related_to")
        })

    import pandas as pd

    pd.DataFrame(node_rows).to_csv(node_list_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edge_list_path, index=False)

    nx.write_graphml(graph, graphml_path)

    with open(json_path, "w") as f:
        json.dump(
            {
                "nodes": node_rows,
                "edges": edge_rows
            },
            f,
            indent=2
        )

    return {
        "nodes": node_list_path,
        "edges": edge_list_path,
        "graphml": graphml_path,
        "json": json_path
    }


def visualize_ontology(graph, output_dir=ONTOLOGY_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        "root": "#bdbdbd",
        "chemical_group": "#9ecae1",
        "pfas_class": "#6baed6",
        "compound": "#2171b5",
        "toxicity_endpoint": "#fb6a4a",
        "risk_level": "#cb181d",
        "exposure_route": "#74c476",
        "environmental_compartment": "#31a354",
        "regulatory_status": "#fdae6b",
        "decision_action": "#756bb1"
    }

    node_colors = [
        color_map.get(graph.nodes[node].get("node_type", "unknown"), "#cccccc")
        for node in graph.nodes()
    ]

    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=0.8,
        iterations=100
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=900,
        node_color=node_colors,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=1.2,
        alpha=0.5
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=8,
        font_weight="bold"
    )

    edge_labels = nx.get_edge_attributes(graph, "relation")

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6
    )

    plt.title("Robust PFAS Ontology Knowledge Graph", fontsize=18)
    plt.axis("off")
    plt.tight_layout()

    png_path = os.path.join(output_dir, "pfas_ontology_visualization.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    return png_path


def create_interactive_ontology_html(graph, output_dir=ONTOLOGY_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis is not installed. Run: pip install pyvis")
        return None

    net = Network(
        height="850px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False
    )

    color_map = {
        "root": "#bdbdbd",
        "chemical_group": "#9ecae1",
        "pfas_class": "#6baed6",
        "compound": "#2171b5",
        "toxicity_endpoint": "#fb6a4a",
        "risk_level": "#cb181d",
        "exposure_route": "#74c476",
        "environmental_compartment": "#31a354",
        "regulatory_status": "#fdae6b",
        "decision_action": "#756bb1"
    }

    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        net.add_node(
            node,
            label=node,
            title=f"{node}<br>Type: {node_type}",
            color=color_map.get(node_type, "#cccccc")
        )

    for source, target, attrs in graph.edges(data=True):
        relation = attrs.get("relation", "related_to")
        net.add_edge(
            source,
            target,
            title=relation,
            label=relation
        )

    net.toggle_physics(True)

    html_path = os.path.join(output_dir, "pfas_ontology_interactive.html")
    net.write_html(html_path)

    return html_path


if __name__ == "__main__":
    graph = build_ontology_graph()

    export_paths = export_ontology(graph)
    png_path = visualize_ontology(graph)
    html_path = create_interactive_ontology_html(graph)

    print("Ontology created successfully.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Static visualization: {png_path}")
    print(f"Interactive visualization: {html_path}")
    print(f"Exports: {export_paths}")

    print("\nExample reasoning path for PFOS:")
    print(get_reasoning_path(graph, "PFOS", "High_Risk"))

    print("\nExample recommended actions for PFOS:")
    print(infer_decision_actions(graph, "PFOS"))