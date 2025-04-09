import os
import random
from androguard.misc import AnalyzeAPK
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

import numpy as np
import pandas as pd


# get sensitive API name from csv 
import csv
def get_API_list(file_path):
    """
    Extract method names from the 'Union' column of a CSV file
    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    list: List containing only method names.
    """
    method_names = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row (e.g., "Restricted API,Highly-Correlation API,...,Union")
        for row in reader:
            if len(row) >= 4:  # Ensure the row has at least 4 columns
                union_column = row[3]  # Fourth column is the Union column (index starts at 0)
                parts = union_column.split(' ')
                if len(parts) > 1:  # Ensure the last column contains a method name
                    method_name = parts[-1]
                    method_names.append(method_name)
    return method_names


def get_permission_list(file_path):
    """
    Read the first row (header) of a CSV file and return the list of column names.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    list: List of column names from the header row (e.g., ["android.permission.GET_ACCOUNTS", "...", "Result"]).
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the first row directly as the header
    return headers

def get_opcodes_list(file_path):
    """
    Read a CSV file and return a list of opcode strings.
    
    Parameters:
    file_path (str): Path to the CSV file 
    Returns:
    list: List of opcode strings      "nop", "move", ...
    """
    opcodes = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row (if exists)
        for row in reader:
            if row:  # Ensure the row is not empty
                opcodes.append(row[0].strip())  # Take the first column and remove whitespace
    return opcodes


# 1. Feature extraction function
def extract_features(apk_path, opcode_list, api_list, perm_list):
    print(f"Starting to process APK file: {apk_path}")
    expected_length = len(opcode_list) + len(api_list) + len(perm_list)
    try:
        a, d, dx = AnalyzeAPK(apk_path)
        features = {}

        # Initialize feature dictionary
        for op in opcode_list:
            features[f"{op}_freq"] = 0
        for api in api_list:
            features[f"{api}_call"] = 0
        for perm in perm_list:
            features[f"{perm}_perm"] = 0

        # Calculate Opcode frequency
        total_instructions = 0
        for method_analysis in dx.get_methods():
            if method_analysis.is_external():
                continue  # Skip external methods as they have no instructions
            encoded_method = method_analysis.get_method()
            if not encoded_method:
                continue
            for inst in encoded_method.get_instructions():
                opcode = inst.get_name()
                if opcode in opcode_list:
                    features[f"{opcode}_freq"] += 1
                total_instructions += 1

        # Detect API calls
        for method_analysis in dx.get_methods():
            encoded_method = method_analysis.get_method()
            if not encoded_method:
                continue
            # Check external methods called by this method
            for _, call, _ in method_analysis.get_xref_to():
                if call.get_name() in api_list:
                    features[f"{call.get_name()}_call"] = 1

        # Check permissions
        permissions = a.get_permissions()
        for perm in perm_list:
            if perm in permissions:
                features[f"{perm}_perm"] = 1

        # Normalize Opcode frequency
        if total_instructions > 0:
            for op in opcode_list:
                features[f"{op}_freq"] /= total_instructions


        feature_values = list(features.values())
        if len(feature_values) != expected_length:
            print(f"Warning: Feature count {len(feature_values)} for sample {apk_path} does not match expected {expected_length}")
            feature_values.extend([0] * (expected_length - len(feature_values)))  # Pad missing parts
        
        
        print(f"Finished processing APK file: {apk_path}")
        return feature_values
    except Exception as e:
        print(f"Error processing {apk_path}: {e}")
        return [0] * expected_length

# 2. Data loading and sampling
def load_and_sample_data(unobf_folders, obf_folders_dict, sample_size=100):
    families = ['adpush', 'artemis', 'dzhtny', 'igexin', 'kuguo', 'leadbolt', 'openconnection', 'spyagent']
    obf_types = ['CallIndirection', 'ClassRename', 'ConstStringEncryption', 'MethodRename']

    # Candidate feature lists
    # opcode_list = ['invoke-virtual', 'move-result-object', 'const-string', 'if-eqz', 'aget-object']
    # api_list = ['sendTextMessage', 'getDeviceId', 'openConnection', 'deleteFile', 'getLine1Number']
    # perm_list = ['android.permission.SEND_SMS', 'android.permission.READ_PHONE_STATE', 'android.permission.INTERNET']
    print("Loading feature lists...")
    opcode_list = get_opcodes_list('/home/xiaowei_3machine/MalScan-master/Data/Dalvik_opcodes.csv')
    api_list = get_API_list('/home/xiaowei_3machine/MalScan-master/Data/API Set.csv')
    perm_list = get_permission_list('/home/xiaowei_3machine/MalScan-master/Data/PermissionList.csv')
    print(f"Feature lists loaded: {len(opcode_list)} opcodes, {len(api_list)} APIs, {len(perm_list)} permissions")
    expected_length = len(opcode_list) + len(api_list) + len(perm_list)
    print(f"Expected feature vector length: {expected_length}")
    
   # Initialize data storage
    data = {'unobf': [], 'obf': {t: [] for t in obf_types}}
    labels = []
    family_samples = defaultdict(list)

    # Load unobfuscated samples
    print("Starting to load unobfuscated samples...")
    for folder in unobf_folders:
        print(f"Processing unobfuscated folder: {folder}")
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.apk'):
                    # Try to match family name from the path
                    family = next((fam for fam in families if fam.lower() in root.lower()), None)
                    if family:
                        family_index = families.index(family)
                        family_samples[family_index].append(os.path.join(root, file))
                    else:
                        print(f"Warning: Could not find family name in path {root}")

    # Sample unobfuscated samples
    for i, fam in enumerate(families):
        total_samples = len(family_samples[i])
        print(f"Family {fam}: Total {total_samples} samples, sampling {min(sample_size, total_samples)}")
        sampled_paths = random.sample(family_samples[i], min(sample_size, total_samples))
        for idx, path in enumerate(sampled_paths, 1):
            print(f"Processing unobfuscated sample {idx}/{len(sampled_paths)}: {path}")
            data['unobf'].append(extract_features(path, opcode_list, api_list, perm_list))
            labels.append(i)


    # Load obfuscated samples
    print("Starting to load obfuscated samples...")
    for family, obf_folders in obf_folders_dict.items():
        family_index = families.index(family)
        family_samples = defaultdict(list)  # Store samples for each obfuscation type
        for folder in obf_folders:
            print(f"Processing obfuscated folder: {folder}")
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.apk'):
                        # Try to match obfuscation type from the path
                        obf_type = next((obf for obf in obf_types if obf.lower() in root.lower()), None)
                        if obf_type:
                            family_samples[obf_type].append(os.path.join(root, file))
                        else:
                            print(f"Warning: Could not find obfuscation type in path {root}")

    # Sample and extract features from obfuscated samples
    for obf_type in obf_types:
        total_samples = len(family_samples[obf_type])
        sample_count = min(sample_size, total_samples)
        print(f"Family {family}, Obfuscation type {obf_type}: Total {total_samples} samples, sampling {sample_count}")
        if total_samples == 0:
            print(f"Warning: No samples for family {family} with obfuscation type {obf_type}, skipping")
            continue
        sampled_paths = random.sample(family_samples[obf_type], sample_count)
        for idx, path in enumerate(sampled_paths, 1):
            print(f"Processing obfuscated sample {idx}/{sample_count}: {path}")
            features = extract_features(path, opcode_list, api_list, perm_list)
            if features:
                data['obf'][obf_type].append(features)
                labels.append(family_index)
            else:
                print(f"Warning: Feature extraction failed for sample {path}")
    # path = '/storage/xiaowei_data/purity_year_data/2017/gap10/benign/AE3AE00DF390B35AF8B4FDB4B3D4958AADE171167F0AEDD950E993A2ADFBE596.apk'
    # features = extract_features(path, opcode_list, api_list, perm_list)   
    # print("feature shape",features.shape)
    # if features:
    #     data['obf'][obf_type].append(features)
    #     labels.append(family_index)
    # else:
    #     print(f"Warning: Feature extraction failed for sample {path}")

    return data, labels, opcode_list, api_list, perm_list


# 3. Dynamic Weighted Feature Selection (DWFS)
def dynamic_weighted_feature_selection(data, labels, opcode_list, api_list, perm_list):
    feature_names = [f"{op}_freq" for op in opcode_list] + \
                    [f"{api}_call" for api in api_list] + \
                    [f"{perm}_perm" for perm in perm_list]
    expected_n_features = len(feature_names)

    # Train on unobfuscated data
    print("Starting to train classifier on unobfuscated data...")
    X_unobf = np.array(data['unobf'])
    n_unobf = len(X_unobf)
    n_features = X_unobf.shape[1]
    print(f"Unobfuscated sample count: {n_unobf}")
    print(f"Unobfuscated feature dimension: {n_features}")
    print(f"Label slice count: {len(labels[:n_unobf])}")
    print(f"Total label count: {len(labels)}")
    if n_features != expected_n_features:
        print(f"Warning: Unobfuscated data feature dimension {n_features} does not match expected {expected_n_features}")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_unobf, labels[:n_unobf])
    I = clf.feature_importances_
    acc_unobf = clf.score(X_unobf, labels[:n_unobf])
    print(f"Unobfuscated data training completed, accuracy: {acc_unobf:.4f}")

    # Stability evaluation on obfuscated data
    obf_types = list(data['obf'].keys())
    alpha = []
    I_obf_list = []
    start_idx = n_unobf
    for obf_type in obf_types:
        X_obf = np.array(data['obf'][obf_type])
        print(f"{obf_type} feature shape: {X_obf.shape}")
        if X_obf.size == 0 or X_obf.ndim != 2:
            print(f"Warning: Data for obfuscation type {obf_type} is empty or malformed, skipping training")
            alpha.append(0)
            I_obf_list.append(np.zeros(n_features))
            continue
        if X_obf.shape[1] != n_features:
            raise ValueError(f"Feature dimension {X_obf.shape[1]} for obfuscation type {obf_type} does not match unobfuscated data {n_features}")
        
        n_samples = len(X_obf)
        print(f"Starting to train classifier for obfuscation type {obf_type}, data shape: {X_obf.shape}")
        print(f"Using label slice: labels[{start_idx}:{start_idx + n_samples}]")
        clf.fit(X_obf, labels[start_idx:start_idx + n_samples])
        acc_obf = clf.score(X_obf, labels[start_idx:start_idx + n_samples])
        alpha.append((acc_unobf - acc_obf) / acc_unobf)
        I_obf_list.append(clf.feature_importances_)
        print(f"Training completed for obfuscation type {obf_type}, accuracy: {acc_obf:.4f}")
        start_idx += n_samples

    # Normalize alpha
    alpha_sum = np.sum(alpha)
    if alpha_sum == 0:
        print("Warning: Sum of alpha is 0, using uniform distribution")
        alpha = np.ones(len(alpha)) / len(alpha)
    else:
        alpha = np.array(alpha) / alpha_sum
    print(f"Alpha weights: {alpha}")

    # Calculate stability change
    print("Calculating feature stability change...")
    delta_I = np.zeros(n_features)
    for j, I_obf in enumerate(I_obf_list):
        delta_I += alpha[j] * np.abs(I - I_obf)

    # Dynamic weights
    beta = 0.2  # Use your improved value
    w1 = 1 - beta * np.mean(alpha)
    w2 = beta * np.mean(alpha)
    print(f"Dynamic weights: w1 = {w1:.4f}, w2 = {w2:.4f}")

    # Comprehensive score
    S = w1 * I - w2 * delta_I
    print(f"Comprehensive score S max value: {np.max(S)}, min value: {np.min(S)}")

    # Create feature dataframe
    feature_data = {
        'opcode': [name if '_freq' in name else '' for name in feature_names[:n_features]],
        'api': [name if '_call' in name else '' for name in feature_names[:n_features]],
        'permissions': [name if '_perm' in name else '' for name in feature_names[:n_features]],
        'score': S,
        'feature_importance_unobf': I,  # Feature importance for unobfuscated data
        'stability_change': delta_I     # Stability change
    }
    df = pd.DataFrame(feature_data)
    
    # Sort by score in descending order and add ranking
    df = df.sort_values(by='score', ascending=False)
    df['ranking'] = range(1, len(df) + 1)

    # Save all features to CSV file
    df.to_csv('features_ranking.csv', index=False)
    print("All features saved to 'features_ranking.csv'")

    # Filter features
    threshold = 0.0005  # Use your improved value
    selected_features = [name for name, score in zip(feature_names[:n_features], S) if score > threshold]
    print(f"Selected features: {selected_features}")

    return selected_features



# Main function
if __name__ == "__main__":
    # Example data paths (replace with your actual paths)
    unobf_folders = [
        '/storage/xiaowei_data/family_data/adpush/',
        '/storage/xiaowei_data/family_data/artemis/',
        '/storage/xiaowei_data/family_data/dzhtny/',
        '/storage/xiaowei_data/family_data/igexin/',
        '/storage/xiaowei_data/family_data/kuguo/',
        '/storage/xiaowei_data/family_data/leadbolt/',
        '/storage/xiaowei_data/family_data/openconnection/',
        '/storage/xiaowei_data/family_data/spyagent/'
    ]
    
    obf_folders_dict = {
        'adpush': [
            '/storage/xiaowei_data/purity_data_obfuscate/adpush/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/adpush/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/adpush/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/adpush/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ],
        'artemis': [
            '/storage/xiaowei_data/purity_data_obfuscate/artemis/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/artemis/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/artemis/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/artemis/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ],
        
        'dzhtny':[
            '/storage/xiaowei_data/purity_data_obfuscate/dzhtny/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/dzhtny/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/dzhtny/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/dzhtny/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ],
        
        'igexin':[
            '/storage/xiaowei_data/purity_data_obfuscate/igexin/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/igexin/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/igexin/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/igexin/Rebuild_NewAlignment_NewSignature_MethodRename/'  
        ],
        
        'kuguo':[
            '/storage/xiaowei_data/purity_data_obfuscate/kuguo/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/kuguo/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/kuguo/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/kuguo/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ],
         
        'leadbolt':[
            '/storage/xiaowei_data/purity_data_obfuscate/leadbolt/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/leadbolt/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/leadbolt/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/leadbolt/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ],
        
        'openconnection':[
            '/storage/xiaowei_data/purity_data_obfuscate/openconnection/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/openconnection/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/openconnection/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/openconnection/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ],
        
        'spyagent':[
            '/storage/xiaowei_data/purity_data_obfuscate/spyagent/Rebuild_NewAlignment_NewSignature_CallIndirection/',
            '/storage/xiaowei_data/purity_data_obfuscate/spyagent/Rebuild_NewAlignment_NewSignature_ClassRename/',
            '/storage/xiaowei_data/purity_data_obfuscate/spyagent/Rebuild_NewAlignment_NewSignature_ConstStringEncryption/',
            '/storage/xiaowei_data/purity_data_obfuscate/spyagent/Rebuild_NewAlignment_NewSignature_MethodRename/'
        ]
     
    }

    # Load and sample data
    print("Loading and sampling data...")
    data, labels, opcode_list, api_list, perm_list = load_and_sample_data(unobf_folders, obf_folders_dict, sample_size=250)
    print("Label distribution:", np.unique(labels, return_counts=True))

    # Perform dynamic weighted feature selection
    print("Performing Dynamic Weighted Feature Selection...")
    selected_features = dynamic_weighted_feature_selection(data, labels, opcode_list, api_list, perm_list)

    # Save results
    with open('selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
    print("Results saved to selected_features.txt")