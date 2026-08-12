"""
Step 1: Data loading.

This script generates a SYNTHETIC dataset that mirrors the real NSL-KDD
schema (same column names, same categorical values, same rough class
imbalance) so the rest of the pipeline can be built and tested today.

TO SWITCH TO YOUR REAL DATA LATER:
  Replace the call to generate_synthetic_data() below with:
      df = pd.read_csv("KDDTrain+.txt", names=COLUMN_NAMES)
  using the real NSL-KDD file you have in your repo. Everything downstream
  (preprocessing, training, evaluation) stays exactly the same, because it
  was written against this same column schema.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# The real NSL-KDD 41 feature names + label (this is the actual published schema)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "telnet", "private", "domain_u", "ftp_data", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO"]

# Real NSL-KDD is heavily imbalanced toward 'normal' -> we mirror that on purpose
LABELS = ["normal"] * 60 + ["dos"] * 25 + ["probe"] * 10 + ["r2l"] * 4 + ["u2r"] * 1


def generate_synthetic_data(n_rows=25000):
    rows = []
    for _ in range(n_rows):
        label = np.random.choice(LABELS)
        is_attack = label != "normal"

        # Attacks tend to have different traffic patterns than normal -> encode that signal
        duration = np.random.exponential(2 if not is_attack else 0.5)
        src_bytes = np.random.exponential(500 if not is_attack else 50)
        dst_bytes = np.random.exponential(1000 if not is_attack else 20)
        num_failed_logins = np.random.poisson(0.1 if not is_attack else 1.5) if label != "r2l" else np.random.poisson(3)
        count = np.random.poisson(5 if not is_attack else 40)  # DoS/probe -> many connections
        serror_rate = np.clip(np.random.normal(0.05 if not is_attack else 0.6, 0.1), 0, 1)

        rows.append({
            "duration": round(duration, 2),
            "protocol_type": np.random.choice(PROTOCOLS, p=[0.7, 0.2, 0.1]),
            "service": np.random.choice(SERVICES),
            "flag": np.random.choice(FLAGS, p=[0.6, 0.15, 0.1, 0.075, 0.075]),
            "src_bytes": int(src_bytes),
            "dst_bytes": int(dst_bytes),
            "land": 0,
            "wrong_fragment": np.random.poisson(0.05),
            "urgent": 0,
            "hot": np.random.poisson(0.2 if not is_attack else 1),
            "num_failed_logins": num_failed_logins,
            "logged_in": np.random.choice([0, 1], p=[0.3, 0.7] if not is_attack else [0.7, 0.3]),
            "num_compromised": np.random.poisson(0.05 if not is_attack else 0.8),
            "root_shell": np.random.choice([0, 1], p=[0.98, 0.02]) if label != "u2r" else np.random.choice([0, 1], p=[0.3, 0.7]),
            "su_attempted": 0,
            "num_root": np.random.poisson(0.05 if not is_attack else 0.5),
            "num_file_creations": np.random.poisson(0.1),
            "num_shells": 0,
            "num_access_files": np.random.poisson(0.1),
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": np.random.choice([0, 1], p=[0.95, 0.05]),
            "count": count,
            "srv_count": np.random.poisson(count * 0.8),
            "serror_rate": round(serror_rate, 2),
            "srv_serror_rate": round(np.clip(serror_rate + np.random.normal(0, 0.05), 0, 1), 2),
            "rerror_rate": round(np.clip(np.random.normal(0.02 if not is_attack else 0.3, 0.05), 0, 1), 2),
            "srv_rerror_rate": round(np.clip(np.random.normal(0.02 if not is_attack else 0.3, 0.05), 0, 1), 2),
            "same_srv_rate": round(np.clip(np.random.normal(0.9 if not is_attack else 0.4, 0.1), 0, 1), 2),
            "diff_srv_rate": round(np.clip(np.random.normal(0.05 if not is_attack else 0.4, 0.1), 0, 1), 2),
            "srv_diff_host_rate": round(np.random.uniform(0, 0.2), 2),
            "dst_host_count": np.random.poisson(50),
            "dst_host_srv_count": np.random.poisson(40),
            "dst_host_same_srv_rate": round(np.clip(np.random.normal(0.8, 0.15), 0, 1), 2),
            "dst_host_diff_srv_rate": round(np.random.uniform(0, 0.3), 2),
            "dst_host_same_src_port_rate": round(np.random.uniform(0, 0.5), 2),
            "dst_host_srv_diff_host_rate": round(np.random.uniform(0, 0.2), 2),
            "dst_host_serror_rate": round(np.clip(serror_rate + np.random.normal(0, 0.05), 0, 1), 2),
            "dst_host_srv_serror_rate": round(np.clip(serror_rate + np.random.normal(0, 0.05), 0, 1), 2),
            "dst_host_rerror_rate": round(np.clip(np.random.normal(0.02 if not is_attack else 0.3, 0.05), 0, 1), 2),
            "dst_host_srv_rerror_rate": round(np.clip(np.random.normal(0.02 if not is_attack else 0.3, 0.05), 0, 1), 2),
            "label": label,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("/home/claude/nids/nids_data.csv", index=False)
    print("Generated:", df.shape)
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nFirst few rows:")
    print(df.head(3))
