import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class NeuralSurrogate(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.trained = False
        self.X = []
        self.y = []

    # ----------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    # ----------------------------------------------------------
    # ✅ NEW FEATURE SET (CRITICAL UPGRADE)
    # ----------------------------------------------------------
    def featurise(self, seq, sfold):
        pd = sfold.get("pair_density") or 0.0
        gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
        length_norm = len(seq) / 50.0

        motif = 1 if ("GAG" in seq or "AAG" in seq) else 0

        energy = sfold.get("cluster_energy_mean")
        prob = sfold.get("cluster_prob_mean")

        if energy is None:
            energy = 0.0
        else:
            energy = max(-50.0, min(0.0, energy)) / -50.0  # normalize

        if prob is None:
            prob = 0.0

        return np.array([
            pd,
            gc,
            length_norm,
            motif,
            energy,
            prob
        ], dtype=np.float32)

    # ----------------------------------------------------------
    def predict_with_uncertainty(self, seq, sfold, n_samples=10):

        fallback = sfold.get("pair_density") or 0.0

        if not self.trained:
            return float(fallback), 0.5

        try:
            x = torch.tensor(
                self.featurise(seq, sfold),
                dtype=torch.float32
            ).to(self.device)

            preds = []

            self.train()  # enable dropout

            for _ in range(n_samples):
                out = self(x)
                preds.append(float(out.item()))

            self.eval()

            if not preds:
                return float(fallback), 1.0

            mean = float(np.mean(preds))
            std = float(np.std(preds))

            # ✅ clamp outputs
            mean = max(0.0, min(1.0, mean))
            std = max(0.01, std)

            return mean, std

        except Exception as e:
            print("Surrogate prediction failed:", e)
            return float(fallback), 0.5

    # ----------------------------------------------------------
    def update(self, data, epochs=30):

        for seq, sf, md in data:
            features = self.featurise(seq, sf)

            if np.any(np.isnan(features)):
                continue

            self.X.append(features)
            self.y.append(md)

        MAX_BUFFER = 500
        if len(self.X) > MAX_BUFFER:
            self.X = self.X[-MAX_BUFFER:]
            self.y = self.y[-MAX_BUFFER:]

        if len(self.X) < 20:
            return

        X = torch.tensor(np.array(self.X), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self.train()

        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        self.trained = True