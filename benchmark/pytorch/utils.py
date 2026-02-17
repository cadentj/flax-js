import chz

from rich.table import Table
from rich.console import Console

def print_topk_table(
    tokens: list[str], probs: list[float], file: str | None = None
) -> None:
    table = Table()
    table.add_column("Token")
    table.add_column("Probability")
    for token, prob in zip(tokens, probs):
        table.add_row(token, f"{prob:.4f}")
    Console().print(table)
    if file:
        with open(file, "w") as f:
            Console(file=f).print(table)

