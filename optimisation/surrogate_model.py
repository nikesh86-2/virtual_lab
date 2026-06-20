import numpy as np
from sklearn.ensemble import RandomForestRegressor


class SurrogateModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50)
        self.X = []
        self.y = []
        self.trained = False

    # ----------------------------------------------------------
    # Feature extraction
    # ----------------------------------------------------------
    def featurise(self, seq, sfold):
        pd = sfold.get("pair_density", 0.0)

        gc = (seq.count("G") + seq.count("C")) / len(seq)

        motif = 0
        if "GAG" in seq or "AAG" in seq:
            motif = 1

        return np.array([pd, gc, len(seq), motif])

    # ----------------------------------------------------------
    # Predict MD score
    # ----------------------------------------------------------
    def predict(self, seq, sfold):
        if not self.trained:
            # fallback heuristic
            return sfold.get("pair_density", 0.0)

        x = self.featurise(seq, sfold).reshape(1, -1)
        return float(self.model.predict(x)[0])

    # ----------------------------------------------------------
    # Update from real MD results
    # ----------------------------------------------------------
    def update(self, data):
        """
        data = list of (seq, sfold, md_score)
        """
        for seq, sf, md in data:
            self.X.append(self.featurise(seq, sf))
            self.y.append(md)

        if len(self.X) >= 10:  # minimum samples
            self.model.fit(self.X, self.y)
            self.trained = True