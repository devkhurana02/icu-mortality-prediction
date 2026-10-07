import pandas as pd
import numpy as np
import re

data = {
    'Email': ['john@example.com', 'abc@', 'xyz.com', np.nan, 'jane@doe.com'],
    'Gender': ['Male', 'male ', 'M', 'FEMALE', 'f'],
    'Transaction_Amount': [100, -500, 250, 999999, 50]
}
df_trans = pd.DataFrame(data)

email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

is_valid_email = df_trans['Email'].str.match(email_pattern, na=False)

df_trans.loc[~is_valid_email, 'Email'] = 'unknown@placeholder.com'

df_trans['Gender'] = df_trans['Gender'].str.strip().str.lower()

gender_map = {'m': 'Male', 'male': 'Male', 'f': 'Female', 'female': 'Female'}
df_trans['Gender'] = df_trans['Gender'].map(gender_map)

print(df_trans)
