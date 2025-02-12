import pandas as pd

def remove_duplicates_and_adjust_targets(X, y, Y_pretrain):
    # Convert to DataFrame
    df_X = pd.DataFrame(X)
    df_y = pd.Series(y, name='target')
    df_Y_pretrain = pd.Series(Y_pretrain, name='target_pretrain')

    # Combine features and targets into a single DataFrame
    df = pd.concat([df_X, df_y, df_Y_pretrain], axis=1)

    # Remove duplicates based on features (X)
    df_unique = df.drop_duplicates(subset=df_X.columns.tolist())

    # Reset index to ensure proper alignment
    df_unique = df_unique.reset_index(drop=True)

    # Separate back into arrays
    X_unique = df_unique[df_X.columns.tolist()].values
    y_unique = df_unique['target'].values
    Y_pretrain_unique = df_unique['target_pretrain'].values

    return X_unique, y_unique, Y_pretrain_unique
