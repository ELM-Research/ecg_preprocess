from utils.file_dir import ensure_directory_exists
from configs.constants import DATA_DIR

def build_base_dataset(args, logger):
    logger.info(f"Building {args.base}")
    if args.base == "mimic_iv":
        from ecg_datasets.base.mimic_iv import MIMIC_IV
        base_dataset_builder = MIMIC_IV(args, logger)
    elif args.base == "ptb_xl":
        from ecg_datasets.base.ptb_xl import PTB_XL
        base_dataset_builder = PTB_XL(args, logger)
    elif args.base == "code15":
        from ecg_datasets.base.code15 import CODE15
        base_dataset_builder = CODE15(args, logger)
    elif args.base == "csn":
        from ecg_datasets.base.csn import CSN
        base_dataset_builder = CSN(args, logger)
    elif args.base == "cpsc":
        from ecg_datasets.base.cpsc import CPSC
        base_dataset_builder = CPSC(args, logger)
    elif args.base == "batch9":
        from ecg_datasets.base.batch9 import BATCH9
        base_dataset_builder = BATCH9(args, logger)
    elif args.base == "batch10":
        from ecg_datasets.base.batch10 import BATCH10
        base_dataset_builder = BATCH10(args, logger)

    if ensure_directory_exists(file = f"{DATA_DIR}/{args.base}/{args.base}.csv"):
        logger.info("DF already exists")
        pass
    else:
        base_dataset_builder.prepare_df()
    
    logger.info("Preparing DF")
    df = base_dataset_builder.get_df()
    base_dataset_builder.create_dataset(df)
