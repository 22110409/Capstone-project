import os
import sys
import argparse
import pandas as pd
import torch
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Client.client import FederatedClient
from Client.server import FederatedServer
from Client.data_utils import clean_dataframe, prepare_for_logistic_regression
from evaluation.evaluation import evaluate_model_on_client_arrays


def infer_input_dim_from_df(df, label_col):
    df = clean_dataframe(df, feature_cols=None)
    X, _, _ = prepare_for_logistic_regression(df, label_col)
    return X.shape[1]


def load_and_prepare_clients(data_dir, label_col, device, balance):

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    excluded = {"df6", "df7", "df10"}
    merge_group = {"df8", "df9"}

    normal_clients = []
    merge_dfs = []

    for f in all_files:
        name = f.lower()

        if any(x in name for x in excluded):
            continue

        df = pd.read_csv(os.path.join(data_dir, f))

        if any(x in name for x in merge_group):
            merge_dfs.append(df)
        else:
            normal_clients.append((f, df))

    clients = []

    for fname, df in normal_clients:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)

        clients.append(
            FederatedClient(
                client_id=f"client_{fname.replace('.csv','')}",
                data_path=tmp.name,
                label_col=label_col,
                device=device,
                balance_strategy=balance,
            )
        )

    if len(merge_dfs) == 2:
        merged_df = pd.concat(merge_dfs, ignore_index=True)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        merged_df.to_csv(tmp.name, index=False)

        clients.append(
            FederatedClient(
                client_id="client_df8_df9_merged_infiltration",
                data_path=tmp.name,
                label_col=label_col,
                device=device,
                balance_strategy=balance,
            )
        )

    return clients



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="Label")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--balance",
        type=str,
        default="class_weight",
        choices=["none", "undersample", "oversample", "class_weight"]
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out_dir", type=str, default="8_results")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clients = load_and_prepare_clients(
        data_dir=args.data_dir,
        label_col=args.label_col,
        device=args.device,
        balance=args.balance,
    )

    print("\n==============================")
    print(" Clients Used for FL Training ")
    print("==============================")
    for c in clients:
        print("-", c.client_id)


    sample_df = pd.read_csv(clients[0].data_path)
    input_dim = infer_input_dim_from_df(sample_df, args.label_col)


    server = FederatedServer(input_dim=input_dim, device=args.device)

    server.train(
        clients=clients,
        rounds=args.rounds,
        epochs=args.epochs,
        lr=args.lr,
    )

    torch.save(server.global_model.state_dict(), "global_model.pt")
    print("\n Training finished. Global model saved as global_model.pt\n")


    print("\n==============================")
    print(" Local Client Model – VALIDATION ")
    print("==============================")

    val_rows = []
    val_clients = clients[:5]

    for c in val_clients:
        _, _, X_val, y_val, _, _ = c.get_train_val_test_tensors(
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        metrics = evaluate_model_on_client_arrays(
            model=c.model,
            X_test=X_val,
            y_test=y_val,
            device=args.device,
        )

        print(
            f"[LOCAL-VAL] {c.client_id}: "
            f"acc={metrics['accuracy']:.4f} | "
            f"prec={metrics['precision']:.4f} | "
            f"rec={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f}"
        )

        val_rows.append({
            "client_id": c.client_id,
            **metrics
        })

    val_path = os.path.join(
        args.out_dir,
        "local_client_validation_metrics_first_5_clients.csv"
    )
    pd.DataFrame(val_rows).to_csv(val_path, index=False)

    print(f"\n Validation metrics saved to: {val_path}\n")


    print("\n==============================")
    print(" Global Model  TEST on First 5 Datasets ")
    print("==============================")

    test_rows = []

    for c in val_clients:
        _, _, X_test, y_test = c.get_train_test_tensors(
            test_size=0.2,
            random_state=42
        )

        metrics = evaluate_model_on_client_arrays(
            model=server.global_model,
            X_test=X_test,
            y_test=y_test,
            device=args.device,
        )

        print(
            f"- {c.client_id}: "
            f"acc={metrics['accuracy']:.4f} | "
            f"prec={metrics['precision']:.4f} | "
            f"rec={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f}"
        )

        test_rows.append({
            "client_id": c.client_id,
            **metrics
        })

    test_path = os.path.join(
        args.out_dir,
        "global_model_test_metrics_first_5_clients.csv"
    )
    pd.DataFrame(test_rows).to_csv(test_path, index=False)

    print(f"\n Test metrics saved to: {test_path}\n")


if __name__ == "__main__":
    main()
