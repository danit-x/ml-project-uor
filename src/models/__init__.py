"""One file per model, one owner per file. Each file exposes build() and train()."""
from . import forest, logreg, svm, tree

MODELS = {
    "Logistic Regression": logreg,
    "Decision Tree": tree,
    "Random Forest": forest,
    "SVM": svm,
}
