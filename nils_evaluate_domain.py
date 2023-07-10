# Imports
from new_resnet_domain_adaption import DomainAdaptationResNet1d
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import tqdm
import h5py
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd
import wfdb
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class ECGDataset1(Dataset):
    """
        Dataset class for ECG data.

        Args:
            data_dir (str): Directory path where the data is stored.
            Transform (torchvision.transforms.Compose): Data transformation to be applied to the ECG data.
    """
    def __init__(self, data_dir, Transform):
        self.transform = Transform
        self.data_dir = data_dir
        self.records = self._load_records()

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
        Retrieves the ECG data and age label for the given index.

        Args:
            index (int): Index of the record.

        Returns:
            torch.Tensor: Processed ECG data.
            int: Age label.
        """
        record = self.records[index]
        
        hea_path = os.path.join(self.data_dir, record + '.hea')
        mat_path = os.path.join(self.data_dir, record )#+ '.mat')

        age = self._extract_age_from_hea(hea_path)
        if age is None:
            age = -100  # Return -100 if age is None
        ecg_data = self._load_ecg_data(mat_path)

        return ecg_data, age

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
        



def compute_loss(ages, pred_ages):
    """
    Computes the loss for age prediction using L1 loss.

    Args:
        ages (torch.Tensor): Ground truth age labels.
        pred_ages (torch.Tensor): Predicted age labels.

    Returns:
        float: The computed Mean Absolute Error (MAE) for age prediction.
    """
    ages = ages.view(-1)
    loss = nn.L1Loss()
    mae = loss(pred_ages, ages)
    return mae

class ECGDataset(Dataset):
        """
        Dataset class for ECG data.

        Args:
            DB_path (str): Database path.
            table_path (str): Path to the table containing the database information.
            Transform (torchvision.transforms.Compose): Data transformation to be applied to the ECG data.
            num_examples (int, optional): Number of examples to consider from the dataset. Default is None (consider all examples).
        """
        def __init__(self, DB_path, table_path, Transform, num_examples = None):
            super().__init__()  # When using a subclass, remember to inherit its properties.
            # ------Your code------#
            # Define self.DB_path, self.table (with pandas reading csv) and self.transform (create an object from the transform we implemented):
            self.DB_path = DB_path
            self.table = pd.read_csv(table_path)
            self.transform = Transform
            self.num_examples = num_examples
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
            Retrieves the ECG data and age label for the given index.

            Args:
                index (int): Index of the record.

            Returns:
                torch.Tensor: Processed ECG data.
                float: Age label.
            """
            # Read the record with wfdb (use get_wfdb_path) and transform its signal. Assign a label by using get_label.
            record = wfdb.rdrecord(self.get_wfdb_path(index))
            signal = self.transform(record.p_signal)  # get tensor with the right dimensions and type.
            label = self.get_label(index)
            # ------^^^^^^^^^------#
            return signal, label

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

class ECGDataset_Domain(Dataset):
    """
    Dataset class for ECG data with domain label.

    Args:
        data_dir (str): Directory path where the data is stored.
        Transform (torchvision.transforms.Compose): Data transformation to be applied to the ECG data.
        domain_label (int): Domain label for the dataset.
    """
    def __init__(self, data_dir, Transform, domain_label):
        self.transform = Transform
        self.data_dir = data_dir
        self.records = self._load_records()
        self.domain_label = domain_label

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
            age = -100  # Return None if age is None
        ecg_data = self._load_ecg_data(mat_path)
  

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
    
class ECGTransform(object):
        """
        Transform the ECG signal into a PyTorch tensor.

        This class applies various transformations to the ECG signal, such as data type conversion and transposition.

        Args:
            signal (numpy.ndarray): ECG signal as a NumPy array.

        Returns:
            torch.Tensor: Transformed ECG signal as a PyTorch tensor.
        """
        def __call__(self, signal):
            #------Your code------#
            # Transform the data type from double (float64) to single (float32) to match the later network weights.
            t_signal = signal.astype(np.single)

            # We transpose the signal to later use the lead dim as the channel... (C,L).
            t_signal = torch.transpose(torch.tensor(t_signal), 0, 1)
            #------^^^^^^^^^------#
            return t_signal  # Make sure I am a PyTorch Tensor
        

def create_confusion_matrix(true_labels_arr, logits_arr, path):
    """
    Create a confusion matrix based on predicted and true age labels.

    Args:
        true_labels_arr (numpy.ndarray): Array of true age labels.
        logits_arr (numpy.ndarray): Array of predicted age labels.
        path (str): Path to save the confusion matrix plot.
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
    plt.savefig(path+'confusionmatrix_test.png')
    #plt.savefig(file_path)

def create_scatter_plot(true_labels_arr, logits_arr, path):
    """
    Create a scatter plot of true age labels versus predicted age labels.

    Args:
        true_labels_arr (numpy.ndarray): Array of true age labels.
        logits_arr (numpy.ndarray): Array of predicted age labels.
        path (str): Path to save the scatter plot.
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
    plt.savefig(path+'scatter_plot_test.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of exams per batch.')
    parser.add_argument('--output', type=str, default='/home/stu15/MachineLearning_ageEstimation/Code/model/evalutation/georgia/',
                        help='output file.')

    parser.add_argument('--ids_dset',
                         help='ids dataset in the hdf5 file.')
    args, unk = parser.parse_known_args()
 
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    device = torch.device('cuda:1')
    # Get checkpoint
    path_to_model = "/home/stu15/MachineLearning_ageEstimation/Code/model"
    ckpt = torch.load(os.path.join(path_to_model, 'model.pth'), map_location=lambda storage, loc: storage)
    # Get config
    config = os.path.join(path_to_model, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = 12
    model = DomainAdaptationResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=1,
                     n_domains = 4,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    test_transform = ECGTransform()
 
    '''ecg_ds1 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g4",Transform=test_transform, domain_label=1)
    ecg_ds2 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g6",Transform=test_transform, domain_label=1)
    ecg_ds3 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g7",Transform=test_transform, domain_label=1)
    ecg_ds = ecg_ds_domain = ConcatDataset([ecg_ds1,ecg_ds2,ecg_ds3])'''

    '''ecg_ds1 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018_extra/g1",Transform=test_transform, domain_label=2)
    ecg_ds2 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018_extra/g2",Transform=test_transform, domain_label=2)
    ecg_ds3 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018_extra/g3",Transform=test_transform, domain_label=2)
    ecg_ds4 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018_extra/g4",Transform=test_transform, domain_label=2)
    ecg_ds = ecg_ds_domain = ConcatDataset([ecg_ds1,ecg_ds2,ecg_ds3,ecg_ds4])'''

    ecg_ds1 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g5",Transform=test_transform, domain_label=2)
    ecg_ds2 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g7",Transform=test_transform, domain_label=2)
    ecg_ds3 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g8",Transform=test_transform, domain_label=2)
    ecg_ds4 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g10",Transform=test_transform, domain_label=2)
    ecg_ds5 = ECGDataset_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g11",Transform=test_transform, domain_label=2)
    ecg_ds = ecg_ds_domain = ConcatDataset([ecg_ds1,ecg_ds2,ecg_ds3,ecg_ds4,ecg_ds5])

    #print(f'The dataset length is {len(ecg_ds)}')
    print('The dataset length is ' + str(len(ecg_ds)))
    batch_size = args.batch_size
    batch_to_show = 10
    ecg_dl = DataLoader(ecg_ds, batch_size=batch_size, shuffle=True, num_workers=0)


    
    n_total = len(ecg_dl)
  
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
        for batch_idx, (X, y, domain_labels) in enumerate(ecg_dl):
                X, y, domain_labels = X.to(device),y.to(device), domain_labels.to(device)
                if y == -100 or y < 18 or y > 89:
                    continue
                X = X.to(device)
                y = y.to(device)
                
                # Forward:
                logits, domain_prediction = model(X, domain_labels)
                
                help = logits
                logits = logits.squeeze()
                if logits.dim() == 0:
                    logits = logits.view(1)
                y = y.float()
                if np.isnan(logits.item()):
                    continue
                loss = compute_loss(y, logits)
                true_labels_list.append(y.cpu().numpy())
                logits_list.append(logits.cpu().numpy())
                
                
                total_loss += loss.detach().cpu().numpy()
                n_entries += 1
    index = range(n_entries)
    total_loss = total_loss/n_entries
    true_labels_arr = np.concatenate(true_labels_list)
    logits_arr = np.concatenate(logits_list)

    create_confusion_matrix(true_labels_arr, logits_arr,args.output)
    #create_scatter_plot(true_labels_arr, logits_arr, args.output)

    
    np.savetxt(args.output+'true_labels_test.csv', true_labels_arr, delimiter=',')
    np.savetxt(args.output+'logits.csv_test', logits_arr, delimiter=',')

    df = pd.DataFrame({'MAE': total_loss}, index = index)
    df = df.set_index('MAE')
    df.to_csv(args.output + 'error_test.csv')
    print("Evaluating done")



