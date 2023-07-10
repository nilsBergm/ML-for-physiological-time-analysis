import json
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from new_resnet_domain_adaption import DomainAdaptationResNet1d
from dataloader import BatchDataloader
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset, ConcatDataset
import torch.optim as optim
import numpy as np
import wfdb
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
import torch.nn.functional as F
import random
from sklearn.metrics import r2_score
import seaborn as sns
import itertools

class ECGTransform(object):
        """
        This will transform the ECG signal into a PyTorch tensor. This is the place to apply other transformations as well, e.g., normalization, etc.
        """
        def __call__(self, signal):
            #------Your code------#
            # Transform the data type from double (float64) to single (float32) to match the later network weights.
            t_signal = signal.astype(np.single)
            t_signal = torch.transpose(torch.tensor(t_signal), 0, 1)
            #------^^^^^^^^^------#
            return t_signal  # Make sure I am a PyTorch Tensor
        
def train_test_split(dataset, ratio, batch_size, num_workers=0):
        """
        :param dataset: PyTorch dataset to split.
        :param ratio: Split ratio for the data - e.g., 0.8 ==> 80% train and 20% test
        :param batch_size: Define the batch_size to use in the DataLoaders.
        :param num_workers: Define the num_workers to use in the DataLoaders.
        :return: train and test DataLoaders.
        """
        # ------Your code------#
        # Hint: You can use torch.randperm to shuffle the data before applying the split.
        '''length = len(dataset)
        random_order = torch.randperm(length)
        N_test = int(round(length*ratio))
        dl_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(random_order[:N_test]), num_workers=num_workers)
        dl_test = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(random_order[N_test:]), num_workers=num_workers)'''
        length = len(dataset)
        N_test = int(round(length * ratio))
        
        # Split the dataset into train and test subsets
        train_dataset = Subset(dataset, range(N_test))
        test_dataset = Subset(dataset, range(N_test, length))
        
        return train_dataset, test_dataset
        # ------^^^^^^^^^------#
        return dl_train, dl_test

def train_Val_test_split(dataset, train_ratio, val_ratio, batch_size, num_workers=0):
    """
    :param dataset: PyTorch dataset to split.
    :param train_ratio: Split ratio for the training data.
    :param val_ratio: Split ratio for the validation data.
    :param batch_size: Define the batch_size to use in the DataLoaders.
    :param num_workers: Define the num_workers to use in the DataLoaders.
    :return: train, validation, and test DataLoaders.
    """
    length = len(dataset)
    N_train = int(round(length * train_ratio))
    N_val = int(round(length * val_ratio))
    
    # Create random order of indices
    random_order = torch.randperm(length)
    
    # Split the dataset into train, validation, and test subsets
    train_indices = random_order[:N_train]
    val_indices = random_order[N_train:N_train + N_val]
    test_indices = random_order[N_train + N_val:]
    
    # Create Subset objects for each subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

def compute_loss_age(ages, pred_ages):
    """
    Computes the loss for age prediction using Mean Absolute Error (MAE).

    Args:
        ages (torch.Tensor): Ground truth age labels.
        pred_ages (torch.Tensor): Predicted age labels.

    Returns:
        float: The computed Mean Absolute Error (MAE) for age prediction.
    """
    ages = ages.view(-1)
    loss = nn.L1Loss()
    mae = 0.0
    count = 0
    for i in range(len(ages)):
        if ages[i]>= 18 and ages[i] <= 89 and not torch.isnan(pred_ages[i]).cpu().numpy():
            mae += loss(pred_ages[i], ages[i])
            count += 1
    if count > 0:
        mae /= count
    else:
        mae = 0
    return mae

def compute_loss_domain(domain_predictions, domain_labels):
    """
    Computes the loss for domain adaptation using Negative Log-Likelihood (NLL) loss.

    Args:
        domain_predictions (torch.Tensor): Predicted domain labels.
        domain_labels (torch.Tensor): Ground truth domain labels.

    Returns:
        float: The computed loss for domain adaptation.
    """
    criterion_domain = nn.NLLLoss()
    domain_loss = 0.0
    count = 0
    for i in range(len(domain_labels)):
        if torch.all(~torch.isnan(domain_predictions[i])) and domain_labels[i] == 0:
            domain_loss += criterion_domain(domain_predictions[i], domain_labels[i])
            count += 1
    if count > 0:
        domain_loss /= count
    else:
        domain_loss = 0
    
    return domain_loss

def train(ep, dataload):
    """
    Trains the model for one epoch.

    Args:
        ep (int): Current epoch number.
        dataload (torch.utils.data.DataLoader): DataLoader for training data.

    Returns:
        float: Total loss for the epoch.
    """
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    
    for batch_idx, (X, y, domain_labels) in enumerate(dataload):
            model.zero_grad()            
            X, y, domain_labels = X.to(device), y.to(device), domain_labels.to(device)
        
            # Forward:
            logits, domain_predictions = model(X, domain_labels)
            #logits = model(X)
            logits = logits.squeeze()
            if logits.dim() == 0:
                logits = logits.view(1)
            y = y.float()
            age_loss = compute_loss_age(y, logits)
            domain_loss = compute_loss_domain(domain_predictions, domain_labels)

            loss = age_loss + domain_loss
            
            if loss == 0:
                continue

            loss.backward()
            # Optimize
            optimizer.step()
            total_loss += loss.detach().cpu().numpy()
            n_entries += 1

            # Update train bar
            train_bar.desc = train_desc.format(ep, total_loss / n_entries)
            train_bar.update(1)


    total_loss = total_loss/n_entries
    train_bar.close()
    return (total_loss), 


def eval(ep, dataload):
    """
    Evaluates the model on the validation dataset for one epoch.

    Args:
        ep (int): Current epoch number.
        dataload (torch.utils.data.DataLoader): DataLoader for validation data.

    Returns:
        float: Total loss for the epoch.
    """
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)

    for batch_idx, (X, y, domain_labels) in enumerate(dataload):
        X, y, domain_labels = X.to(device),y.to(device), domain_labels.to(device)
        with torch.no_grad():
            # Forward pass
            logits, domain_predictions = model(X, domain_labels)
            logits = logits.squeeze()
            y = y.float()
            if logits.dim() == 0:
                logits = logits.view(1)

            age_loss = compute_loss_age(y, logits)
            loss_domain = compute_loss_domain(domain_predictions, domain_labels)

            loss = age_loss + loss_domain
            # Update ids
            if loss == 0:
                total_loss += loss
            else:
                total_loss += loss.detach().cpu().numpy()
            n_entries +=1
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    total_loss = total_loss/n_entries
    return total_loss 



def plot_loss(train_loss_list, valid_loss_list, folder, epochs):
    """
    Plots the train and validation losses over epochs.

    Args:
        train_loss_list (list): List of train losses for each epoch.
        valid_loss_list (list): List of validation losses for each epoch.
        folder (str): Folder path to save the plot.
        epochs (int): Total number of epochs.

    Returns:
        None
    """
    epochs = range(0, epochs)

    plt.plot(epochs, train_loss_list, label='Train MAE')
    plt.plot(epochs, valid_loss_list, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Train and Validation MAE')
    plt.legend()
    plt.savefig(os.path.join(folder, 'mae_plot.png'))

def create_scatter_plot(true_labels_arr, logits_arr, path):
    """
    Creates a scatter plot of true age versus predicted age.

    Args:
        true_labels_arr (np.ndarray): Array of true age labels.
        logits_arr (np.ndarray): Array of predicted age logits.
        path (str): Path to save the scatter plot.

    Returns:
        None
    """
    
    # Create a scatter plot
    plt.scatter(logits_arr, true_labels_arr, color='blue', label='Data Points')

    # Add a diagonal line
    x = np.linspace(np.min(logits_arr), np.max(true_labels_arr), 100)
    plt.plot(x, x, color='red', label='perfect value')

    # Set labels and title
    plt.xlabel('Logits')
    plt.ylabel('True Age')
    plt.title('True age vs. predicted age')
    # Add legend
    plt.legend()

    # Save the scatter plot as an image
    plt.savefig(path+'in_distri_scatter_plot_test.png')

def create_confusion_matrix(true_labels_arr, logits_arr, path):
    """
    Creates a confusion matrix based on age intervals.

    Args:
        true_labels_arr (np.ndarray): Array of true age labels.
        logits_arr (np.ndarray): Array of predicted age logits.
        path (str): Path to save the confusion matrix.

    Returns:
        None
    """
    intervall = 10
    age_intervals = range(10, 100, intervall)

    # Gruppiere die vorhergesagten Alter in Intervalle
    predicted_age_intervals = np.digitize(np.round(logits_arr), age_intervals)

    # Gruppiere die wahren Alterskategorien in Intervalle
    true_age_intervals = np.digitize(true_labels_arr, age_intervals)

    # Berechne die Confusion Matrix basierend auf den Altersintervallen
    cm = confusion_matrix(true_age_intervals, predicted_age_intervals)
    # Calculate the confusion matrix

    # Plot the confusion matrix with age interval labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    # Define the tick positions and labels for x-axis
    x_tick_positions = np.arange(len(age_intervals)) + 0.5
    x_tick_labels = [f'{age}-{age + intervall}' for age in age_intervals]

    # Define the tick positions and labels for y-axis
    y_tick_positions = np.flip(np.arange(len(age_intervals))) + 0.5
    y_tick_labels = np.flip([f'{age}-{age + intervall}' for age in age_intervals])

    # Set the tick positions and labels for x-axis and y-axis
    plt.xticks(x_tick_positions, x_tick_labels, rotation='vertical')
    plt.yticks(y_tick_positions, y_tick_labels)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted age')
    plt.ylabel('True age')
    file_path = os.path.join('model', 'evaluation', 'confusionmatrix_test.png')
    plt.savefig(path+'in_distri_confusionmatrix_test.png')
    #plt.savefig(file_path)
    
class ECGDataset_Domain(Dataset):
    """
    Dataset class for ECG data with domain adaptation.

    Args:
        data_dir (str): Directory path where the data is stored.
        Transform (torchvision.transforms.Compose): Data transformation to be applied to the ECG data.
        domain_label (int): Domain label for domain adaptation.
        known_label (bool, optional): Flag indicating whether the age labels are known or not. Default is False.
    """
    def __init__(self, data_dir, Transform, domain_label, known_label = False):
        self.transform = Transform
        self.data_dir = data_dir
        self.records = self._load_records()
        self.domain_label = domain_label
        self.known_label = known_label

    def _load_records(self):
        """
        Loads the records from the RECORDS file.

        Returns:
            list: List of record names.
        """
        records_path = os.path.join(self.data_dir, 'RECORDS')
        with open(records_path, 'r') as f:
            records = [line.strip() for line in f]
        return records

    def __len__(self):
        """
        Returns the number of records in the dataset.

        Returns:
            int: Number of records.
        """
        return len(self.records)

    def __getitem__(self, index):
        """
        Retrieves the ECG data, age label, and domain label for the given index.

        Args:
            index (int): Index of the record.

        Returns:
            torch.Tensor: Processed ECG data.
            int: Age label.
            int: Domain label.
        """
        record = self.records[index]
        
        hea_path = os.path.join(self.data_dir, record + '.hea')
        mat_path = os.path.join(self.data_dir, record )#+ '.mat')

        age = self._extract_age_from_hea(hea_path)
        if age is None:
            age = -100  # Return -100 if age is None
        ecg_data = self._load_ecg_data(mat_path)
        if self.known_label == False:
            age = -100 #I deal like I dont know the true label

        return ecg_data, age, self.domain_label

    def _extract_age_from_hea(self, hea_path):
        """
        Extracts the age label from the .hea file.

        Args:
            hea_path (str): Path to the .hea file.

        Returns:
            int or None: Extracted age label or None if age is missing or invalid.
        """
        age = None
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('# Age:'):
                    age_string = line.split(':')[1].strip()
                    if age_string != "NaN":
                        age = int(line.split(':')[1].strip())
                    else:
                        age = None
                    break
        
        return age

    def _load_ecg_data(self, mat_path):
        """
        Loads the ECG data from the .mat file and processes it.

        Args:
            mat_path (str): Path to the .mat file.

        Returns:
            torch.Tensor: Processed ECG data.
        """
        record = wfdb.rdrecord(mat_path)
        signals = record.p_signal
        num_leads = signals.shape[1]
        processed_signals = []

        for lead in range(num_leads):
            signal = signals[:, lead]
            
            # Ensure the signal has a length of 1000
            if len(signal) < 1000:
                # Pad the signal with zeros if it's shorter than 1000
                padding = np.zeros(1000 - len(signal))
                signal = np.concatenate((signal, padding))
            elif len(signal) > 1000:
                # Truncate the signal if it's longer than 1000
                signal = signal[:1000]
            
            processed_signals.append(signal)
        processed_signals = self.transform(np.array(processed_signals))
        #return processed_signals
     
        return  torch.transpose(processed_signals, 0, 1)
class ECGDataset(Dataset):
        """
        Dataset class for ECG data.

        Args:
            DB_path (str): Database path.
            table_path (str): Path to the table containing the database information.
            Transform (torchvision.transforms.Compose): Data transformation to be applied to the ECG data.
            domain_label (int): Domain label for the dataset.
            num_examples (int, optional): Number of examples to consider from the dataset. Default is None (consider all examples).
        """
        def __init__(self, DB_path, table_path, Transform,domain_label, num_examples = None):
            super().__init__()  # When using a subclass, remember to inherit its properties.
            # ------Your code------#
            # Define self.DB_path, self.table (with pandas reading csv) and self.transform (create an object from the transform we implemented):
            self.DB_path = DB_path
            self.table = pd.read_csv(table_path)
            self.transform = Transform
            self.num_examples = num_examples
            self.domain_label = domain_label
            # ------^^^^^^^^^------#

        def get_wfdb_path(self, index):
            """
            Retrieves the wfdb path for the given index.

            Args:
                index (int): Index of the record.

            Returns:
                str: Wfdb path.
            """

            # Get the wfdb path as given in the database table:
            wfdb_path = self.DB_path + table['filename_lr'][int(index)]
            return wfdb_path

        def get_label(self, index):
            """
                Determines the label for the given index.

            Args:
                    index (int): Index of the record.

                Returns:
                    float: Age label.

            """
            # A method to decide the label:
            age_str = self.table["age"][int(index)]
            age_float = float(age_str)
            return age_float
            

        def __getitem__(self, index):
            """
            Retrieves the ECG data, age label, and domain label for the given index.

            Args:
                index (int): Index of the record.

            Returns:
                torch.Tensor: Processed ECG data.
                int: Age label.
                int: Domain label.
            """
            # Read the record with wfdb (use get_wfdb_path) and transform its signal. Assign a label by using get_label.
            record = wfdb.rdrecord(self.get_wfdb_path(index))
            signal = self.transform(record.p_signal)  # get tensor with the right dimensions and type.
            label = self.get_label(index)
            # ------^^^^^^^^^------#
            return signal, label, self.domain_label

        def __len__(self):
            """
            Returns the number of records in the dataset.

            Returns:
                int: Number of records.
            """
            if self.num_examples is not None:
                return min(len(self.table), self.num_examples)
            else:
                return len(self.table)



if __name__ == "__main__":
    #import h5py
    import pandas as pd
    import argparse
    from warnings import warn
    random.seed(42)

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=1000,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=1, #default war 10 NB
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32, old = 64).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001, old = 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.2,
                        help='reducing factor for the lr in a plateu (default: 0.1, old 0.2)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 450],#[64, 128, 128, 128, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[1000, 500,250, 125,25],#[5000, 1000, 200, 40, 8],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='dropout rate (default: 0.8, 0.8).')
    parser.add_argument('--kernel_size', type=int, default=19,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')

    parser.add_argument('--cuda', action='store_true',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training')
    parser.add_argument('--output', type=str, default='/home/stu15/MachineLearning_ageEstimation/Code/model/evalutation/',
                        help='output file.')
 
    args, unk = parser.parse_known_args()


    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    # Set device
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device('cuda:1')
    print(device)

    folder = args.folder

    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # Save config file
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    tqdm.write("Building data loaders...")
  
    DB_path = '/home/stu15/MachineLearning_ageEstimation/Code/ptb-xl-a-large/'
    table_path = DB_path + 'ptbxl_database_new.csv'
    table = pd.read_csv(table_path)
    ECG_path = DB_path+'records100/'  # we will use the 100Hz recording version.
    list_of_folders = os.listdir(ECG_path)
    list_of_files = os.listdir(ECG_path+list_of_folders[0])

    record = wfdb.rdrecord(ECG_path + list_of_folders[0] + '/' + list_of_files[0][:-4])
    fs = record.fs
    lead_names = record.sig_name


    signal = record.p_signal
    test_transform = ECGTransform()
    transformed = test_transform(signal)  # __call__ method

    ecg_ds =  ECGDataset(DB_path, table_path, ECGTransform(), domain_label = 0)

    

    #print(f'The dataset length is {len(ecg_ds)}')
    print('The dataset length is ' + str(len(ecg_ds)))
    batch_size = args.batch_size
   


    #Target Domain dataset
    target_transform = ECGTransform()
 
    ecg_ds_domain1 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g1",Transform=target_transform, domain_label=1,known_label=True)
    ecg_ds_domain2 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g2",Transform=target_transform, domain_label=1,known_label=True)
    ecg_ds_domain3 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g3",Transform=target_transform, domain_label=1,known_label=False)
    ecg_ds_domain4 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g4",Transform=target_transform, domain_label=1,known_label=False)
    ecg_ds_domain5 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g5",Transform=target_transform, domain_label=1,known_label=False)

    ecg_ds_domain6 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g1",Transform=target_transform, domain_label=2,known_label=True)
    ecg_ds_domain7 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g2",Transform=target_transform, domain_label=2,known_label=True)
    ecg_ds_domain8 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g3",Transform=target_transform, domain_label=2,known_label=False)
    ecg_ds_domain9 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g4",Transform=target_transform, domain_label=2,known_label=False)
    ecg_ds_domain10 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g5",Transform=target_transform, domain_label=2,known_label=False)
    ecg_ds_domain11 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g6",Transform=target_transform, domain_label=2,known_label=False)
    ecg_ds_domain12 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g7",Transform=target_transform, domain_label=2,known_label=False)
    ecg_ds_domain13 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g8",Transform=target_transform, domain_label=2,known_label=False)
    ecg_ds_domain14 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g9",Transform=target_transform, domain_label=2,known_label=False)


    ecg_ds_domain15 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/ptb/g1",Transform=target_transform, domain_label=3,known_label=False)

    batch_size = args.batch_size
    batch_to_show = 10
    #ecg_dl_domain = DataLoader(ecg_ds_domain, batch_size=batch_size, shuffle=True, num_workers=0)


    #ecg_ds_domain = ConcatDataset([ecg_ds_domain1, ecg_ds_domain2, ecg_ds_domain3, ecg_ds_domain4, ecg_ds_domain5, ecg_ds_domain6, ecg_ds_domain7, ecg_ds_domain8, ecg_ds_domain9, ecg_ds_domain10, ecg_ds_domain11, ecg_ds_domain12, ecg_ds_domain13, ecg_ds_domain14, ecg_ds_domain15])
    ecg_ds_domain = ConcatDataset([ecg_ds_domain1, ecg_ds_domain2, ecg_ds_domain3, ecg_ds_domain5, ecg_ds_domain6, ecg_ds_domain7, ecg_ds_domain8, ecg_ds_domain9, ecg_ds_domain11, ecg_ds_domain14, ecg_ds_domain15])
    # Create a list of indices
    indices = list(range(len(ecg_ds_domain)))

    # Shuffle the indices
    random.shuffle(indices)

    # Create a new shuffled dataset using the shuffled indices
    ecg_ds_domain = Subset(ecg_ds_domain, indices)

    ds_train, ds_val, ds_test = train_Val_test_split(ecg_ds, train_ratio=0.80, val_ratio = 0.1,  batch_size=batch_size) 
    ds_train_domain, ds_val_domain = train_test_split(ecg_ds_domain, ratio = 0.9, batch_size = batch_size) 
    
    combined_ds_train = ConcatDataset([ds_train, ds_train_domain ])
    combined_ds_val = ConcatDataset([ds_val, ds_val_domain])

    # Create a new dataloader from the shuffled list
    combined_dl_train = DataLoader(combined_ds_train, batch_size=batch_size, shuffle=True)
    combined_dl_val = DataLoader(combined_ds_val, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
    #end myCode
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_Output = 1#len(labels_train_set)  # just the age
    print("Number of outputs")
    print(N_Output)
    #print(labels_train_set)
    model = DomainAdaptationResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_Output,
                     n_domains= 4,
                     kernel_size=args.kernel_size,
                     dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    weight_decay = 0.001
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=weight_decay)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    
    train_MAE_list = []
    valid_MAE_list = []

    epochs = 0

    for ep in range(start_epoch, args.epochs):
        train_loss  = train(ep, combined_dl_train)
        valid_loss = eval(ep, combined_dl_val)
        train_MAE_list.append(train_loss)
        valid_MAE_list.append(valid_loss)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            epochs += 1
            break
        # Save history
        new_row = {"epoch": ep, "train_mae": train_loss, "valid_mae": valid_loss, "lr": learning_rate}
        history = history.append(new_row, ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
        epochs +=1
    
    plot_loss(train_MAE_list, valid_MAE_list, folder,epochs)
    tqdm.write("Done!")


    ###In distribution test
    n_total = len(dl_test)
    predicted_age = np.zeros((n_total,))
    # Evaluate on test data
    model.eval()
    n_batches = int(np.ceil(n_total/args.batch_size))
    # Compute gradients
    predicted_age = np.zeros((n_total,))
    end = 0

    total_loss = 0
    n_entries = 0
    true_labels_list = []
    logits_list = []
    with torch.no_grad():
        for X, y, domain_label in dl_test:
                X = X.to(device)
                y = y.to(device)
                domain_label = domain_label.to(device)


                
                
                # Forward:
                logits, domain_pred = model(X,domain_label)
                logits = logits.squeeze()
                if logits.dim() == 0:
                    logits = logits.view(1)
                y = y.float()
                loss = compute_loss_age(y, logits)
                true_labels_list.append(y.cpu().numpy())
                logits_list.append(logits.cpu().numpy())
                
                total_loss += loss.detach().cpu().numpy()
                n_entries += 1

    index = range(n_entries) 
    total_loss = total_loss/n_entries
    true_labels_arr = np.concatenate(true_labels_list)
    logits_arr = np.concatenate(logits_list)

    create_scatter_plot(true_labels_arr, logits_arr, args.output)

    
    np.savetxt(args.output+'in_distri_true_labels_test.csv', true_labels_arr, delimiter=',')
    np.savetxt(args.output+'in_distri_logits.csv_test', logits_arr, delimiter=',')

    df = pd.DataFrame({'MAE': total_loss}, index = index)
    df = df.set_index('MAE')
    df.to_csv(args.output + 'in_distri_error_test.csv')
    print("Evaluating done")
  



    def find_besthyper():
         # Define the range or set of values for each hyperparameter
        batch_size_range = [32,64, 128, 256]
        lr_range = [0.001, 0.01, 0.1, 0.0001]
        min_lr_range = [1e-5, 1e-4, 1e-3]
        lr_factor_range = [0.1, 0.2, 0.5]
        dropout_rate_range = [0.2, 0.4, 0.6]
        kernel_size_range = [15, 17, 19, 25]

        # Define the number of random search iterations
        num_iterations = 30

        best_loss = float('inf')
        best_hyperparameters = {}

        for iteration in range(num_iterations):
            # Arguments that will be saved in config file
            parser = argparse.ArgumentParser(add_help=True,
                                            description='Train model to predict rage from the raw ecg tracing.')
            parser.add_argument('--epochs', type=int, default=70,
                                help='maximum number of epochs (default: 70)')
            parser.add_argument('--seed', type=int, default=2,
                                help='random seed for number generator (default: 2)')
            parser.add_argument('--sample_freq', type=int, default=100,
                                help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
            parser.add_argument('--seq_length', type=int, default=1000,
                                help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                            'to fit into the given size. (default: 4096)')
            parser.add_argument('--scale_multiplier', type=int, default=1, #default war 10 NB
                                help='multiplicative factor used to rescale inputs.')
            parser.add_argument('--batch_size', type=int, default=128,
                                help='batch size (default: 32).')
            parser.add_argument('--lr', type=float, default=0.001,
                                help='learning rate (default: 0.001)')
            parser.add_argument("--patience", type=int, default=7,
                                help='maximum number of epochs without reducing the learning rate (default: 7)')
            parser.add_argument("--min_lr", type=float, default=1e-5,
                                help='minimum learning rate (default: 1e-7)')
            parser.add_argument("--lr_factor", type=float, default=0.1,
                                help='reducing factor for the lr in a plateu (default: 0.1)')
            parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 450],#[64, 128, 128, 128, 320],
                                help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
            parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[1000, 500,250, 125,25],#[5000, 1000, 200, 40, 8],
                                help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
            parser.add_argument('--dropout_rate', type=float, default=0.4,
                                help='dropout rate (default: 0.8).')
            parser.add_argument('--kernel_size', type=int, default=17,
                                help='kernel size in convolutional layers (default: 17).')
            parser.add_argument('--folder', default='model/',
                                help='output folder (default: ./out)')

            parser.add_argument('--cuda', action='store_true',
                                help='use cuda for computations. (default: False)')
            parser.add_argument('--n_valid', type=int, default=100,
                                help='the first `n_valid` exams in the hdf will be for validation.'
                                    'The rest is for training')
        
            args, unk = parser.parse_known_args()
            # Randomly sample hyperparameters from their defined ranges
            batch_size = random.choice(batch_size_range)
            lr = random.choice(lr_range)
            min_lr = random.choice(min_lr_range)
            lr_factor = random.choice(lr_factor_range)
            dropout_rate = random.choice(dropout_rate_range)
            kernel_size = random.choice(kernel_size_range)
            
            # Set the hyperparameters
            args.batch_size = batch_size
            args.lr = lr
            args.min_lr = min_lr
            args.lr_factor = lr_factor
            args.dropout_rate = dropout_rate
            args.kernel_size = kernel_size
            

            # Check for unknown options
            if unk:
                warn("Unknown arguments:" + str(unk) + ".")

            torch.manual_seed(args.seed)
            # Set device

            device = torch.device('cuda:0')
            print(device)

            folder = args.folder

            # Generate output folder if needed
            if not os.path.exists(args.folder):
                os.makedirs(args.folder)
            # Save config file
            with open(os.path.join(args.folder, 'args.json'), 'w') as f:
                json.dump(vars(args), f, indent='\t')

            tqdm.write("Building data loaders...")
        
            DB_path = '/home/stu15/MachineLearning_ageEstimation/Code/ptb-xl-a-large/'
            table_path = DB_path + 'ptbxl_database_new.csv'
            table = pd.read_csv(table_path)
            ECG_path = DB_path+'records100/'  # we will use the 100Hz recording version.
            list_of_folders = os.listdir(ECG_path)
            list_of_files = os.listdir(ECG_path+list_of_folders[0])

            record = wfdb.rdrecord(ECG_path + list_of_folders[0] + '/' + list_of_files[0][:-4])
            fs = record.fs
            lead_names = record.sig_name


            signal = record.p_signal
            test_transform = ECGTransform()
            transformed = test_transform(signal)  # __call__ method

            ecg_ds =  ECGDataset(DB_path, table_path, ECGTransform(), num_examples = 10000)

            

            #print(f'The dataset length is {len(ecg_ds)}')
            print('The dataset length is ' + str(len(ecg_ds)))
            batch_size = args.batch_size
        
            ecg_dl = DataLoader(ecg_ds, batch_size=batch_size, shuffle=True, num_workers=0)


            dl_train, dl_test = train_test_split(ecg_ds, ratio=0.90, batch_size=batch_size) 

            #end myCode
            tqdm.write("Done!")

            tqdm.write("Define model...")
            N_LEADS = 12  # the 12 leads
            N_Output = 1#len(labels_train_set)  # just the age
            print("Number of outputs")
            print(N_Output)
            #print(labels_train_set)
            model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                            blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                            n_classes=N_Output,
                            kernel_size=args.kernel_size,
                            dropout_rate=args.dropout_rate)
            model.to(device=device)
            tqdm.write("Done!")

            tqdm.write("Define optimizer...")
            optimizer = optim.Adam(model.parameters(), args.lr)
            tqdm.write("Done!")

            tqdm.write("Define scheduler...")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                            min_lr=args.lr_factor * args.min_lr,
                                                            factor=args.lr_factor)
                
            # Perform training and evaluation with the current hyperparameters
            train_loss  = train(iteration, dl_train)
            valid_MAE = eval(iteration, dl_test)
            #train_MAE, valid_MAE = train_and_evaluate(args)
            
            # Check if the current hyperparameters are the best
            if valid_MAE < best_loss:
                best_loss = valid_MAE
                best_hyperparameters = {
                    'batch_size': batch_size,
                    'lr': lr,
                    'min_lr': min_lr,
                    'lr_factor': lr_factor,
                    'dropout_rate': dropout_rate,
                    'kernel_size': kernel_size
                }

        # Print the best hyperparameters
        print("Best Hyperparameters:")
        print(best_hyperparameters)



