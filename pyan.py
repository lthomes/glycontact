import ast
import graphviz as gv

def extract_functions(tree):
    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = set()
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                    functions[node.name].add(subnode.func.id)

    return functions

def draw_dependency_graph(functions):
    dot = gv.Digraph(comment='Dependency Graph')

    for name, calls in functions.items():
        dot.node(name)
        for call in calls:
            if call != name:
                dot.edge(name, call)

    return dot

def generate_dependency_graph(filename):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    functions = extract_functions(tree)
    dot = draw_dependency_graph(functions)
    return dot

if __name__ == "__main__":
    # Remplacez 'your_file.py' par le nom de votre fichier Python
    filename = "process.py"

    dependency_graph = generate_dependency_graph(filename)
    dependency_graph.render("dependency_graph.png", view=True)
