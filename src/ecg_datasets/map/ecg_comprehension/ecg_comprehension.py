from tqdm import tqdm
import glob
import string

from ecg_datasets.map.map_dataset import SyntheticDataset
from configs.constants import DATA_DIR
from utils.file_dir import ensure_directory_exists

class ECGComprehension(SyntheticDataset):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.data_name = "mimic_iv"
        self.saved_dir = glob.glob(f"{DATA_DIR}/{self.data_name}/preprocessed_{self.args.segment_len}/*.npy")
        ecg_comprehensions = "_".join(self.args.ecg_comprehension)
        self.save_dir_json = f"src/ecg_datasets/map/ecg_comprehension/{self.args.map}_{ecg_comprehensions}_{self.args.ecg_comprehension_per_len}_hf.json"
        self.saved_dir

    def get_map_data(self,):
        if ensure_directory_exists(file=self.save_dir_json):
            print("ECG Comprehension json already exists")
        else:
            data = self.create_json()
        return data

    def create_json(self):
        data = []
        # to ensure balance
        total_len = len(self.args.ecg_comprehension) * self.args.ecg_comprehension_per_len
        print("total len", total_len)
        prompt = "What type of signal is provided?"
        question, mapping = self.format_mcq_question(prompt, self.args.ecg_comprehension)
        print("question", question)
        print("mapping", mapping)
        for i in tqdm(range(self.args.ecg_comprehension_per_len)):
            for signal_type in self.args.ecg_comprehension:
                if signal_type == "ecg":
                    ecg_path = self.saved_dir[i]
                else:
                    ecg_path = signal_type
                text = [{"from": "human", "value": question},
                        {"from": "gpt", "value": mapping[signal_type]}]
                name = signal_type
                data.append({
                    "ecg_path": ecg_path,
                    "text": text,
                    "name": name
                })
        return data

    def format_mcq_question(self, prompt: str, choices: list[str]) -> str:
        labels = string.ascii_uppercase[:len(choices)]
        mapping = dict(zip(choices, labels))
        question = prompt + "\n" + "\n".join(f"{k}) {v}" for k, v in mapping.items())
        return question, mapping
