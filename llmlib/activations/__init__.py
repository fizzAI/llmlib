from gelu import GELU, ApproximateGELU

def get_activation_by_name(name: str):
    match name:
        case "gelu":
            return GELU()
        case "approx_gelu":
            return ApproximateGELU()
        case _:
            raise ValueError(f"Unknown activation function: {name}")