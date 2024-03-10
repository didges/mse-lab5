import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    def _tree_traversal(tree, index=0):
        if tree.tree_.children_left[index] == -1 and tree.tree_.children_right[index] == -1:
            return {"class": int(tree.tree_.value[index].argmax())}
        return {
            "feature_index": int(tree.tree_.feature[index]),
            "threshold": round(float(tree.tree_.threshold[index]), 4),
            "left": _tree_traversal(tree, tree.tree_.children_left[index]),
            "right": _tree_traversal(tree, tree.tree_.children_right[index])
        }
    return json.dumps(_tree_traversal(tree))

def generate_sql_query(tree_as_json: str, features: list) -> str:
    data = json.loads(tree_as_json)

    def _sub_generate(json_data: dict):

        if 'left' not in json_data.keys() and 'right' not in json_data.keys():
            class_leaf = json_data['class']
            return f"{class_leaf}"

        elif 'left' in json_data.keys() and 'class' in json_data['right'].keys():
            feature_name = features[int(json_data['feature_index'])]
            threshold = float(json_data['threshold'])
            return f"CASE\n\t WHEN {feature_name} > {threshold} THEN \n {json_data['right']['class']} ELSE \n {_sub_generate(json_data['left'])} END"

        elif 'right' in json_data.keys() and 'class' in json_data['left'].keys():
            feature_name = features[int(json_data['feature_index'])]
            threshold = float(json_data['threshold'])
            return f"CASE \n\t WHEN {feature_name} > {threshold} THEN \n {_sub_generate(json_data['right'])} ELSE \n {json_data['left']['class']} END"

        elif 'left' in json_data.keys() and 'right' in json_data.keys():
            feature_name = features[int(json_data['feature_index'])]
            threshold = float(json_data['threshold'])
            sub_sql = f"CASE \n\t WHEN {feature_name} > {threshold} THEN \n"
            return f"{sub_sql} {_sub_generate(json_data['right'])} ELSE \n {_sub_generate(json_data['left'])} END"

    return f"SELECT {_sub_generate(data)} AS class_label"


if __name__ == "__main__":
    X, y = make_classification(n_samples=100,
                               n_features=5, n_informative=3, n_classes=2)
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)

    features = ['f0', 'f1', 'f2', 'f3', 'f4']
    some_json = convert_tree_to_json(model)
    print(generate_sql_query(some_json, features))
