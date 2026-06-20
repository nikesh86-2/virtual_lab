
def generate_objectives(parsed: dict) -> dict:
    extra = {}

    # weak binding → increase binding pressure
    if parsed["dg_best"] and parsed["dg_best"] > -7:
        extra["binding"] = 0.5

    # high spread → convergence instability
    if parsed["spread"] and parsed["spread"] > 5:
        extra["structure"] = 0.3

    # missing MD → stability proxy
    if "MD trajectory" in parsed["missing_controls"]:
        extra["kinetic"] = 0.4

    # entropy missing → ensemble diversity
    if any("entropy" in c.lower() for c in parsed["concerns"]):
        extra["structure"] = extra.get("structure", 0) + 0.2

    return extra
