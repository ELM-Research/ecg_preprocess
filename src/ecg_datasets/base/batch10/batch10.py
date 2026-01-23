
import xml.etree.ElementTree as ET
import glob
import pandas as pd
from pathlib import Path

from configs.constants import DATA_DIR, PTB_ORDER

from ecg_datasets.base.base_dataset import BaseDataset

from utils.viz import plot_ecg
from utils.batchn import parse_report, parse_sf, parse_ecg

class BATCH10(BaseDataset):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def prepare_df(self, ):
        self.logger.info("Preparing DF")
        xml_paths = glob.glob(f"{DATA_DIR}/{self.args.base}/*.xml")
        pd.DataFrame({"xml_path": xml_paths}).to_csv(f"{DATA_DIR}/{self.args.base}/{self.args.base}.csv", index=False)

    def open_ecg(self, row,):
        xml_path = row["xml_path"]
        root = ET.parse(xml_path).getroot()
        report = parse_report(root)
        sf = parse_sf(root)
        ecg = parse_ecg(root)
        assert sf == 250 and ecg.shape == (2500, 12)
        # plot_ecg(ecg, PTB_ORDER, title = Path(xml_path).stem)
        return {"file_path": xml_path, "sf": sf,
                "report": report, "file_name": Path(xml_path).stem,
                "ecg": ecg,}
