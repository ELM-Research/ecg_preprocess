def build_map_dataset(args, logger):
    logger.info(f"Building {args.map}")
    if args.map == "pretrain_mimic":
        from ecg_datasets.map.pretrain_mimic import PretrainMIMIC
        map_dataset_builder = PretrainMIMIC(args, logger)
    elif args.map == "ecg_grounding":
        from ecg_datasets.map.ecg_grounding import ECGGrounding
        map_dataset_builder = ECGGrounding(args, logger)
    elif args.map == "ecg_qa_mimic_iv" or args.map == "ecg_qa_ptb_xl":
        from ecg_datasets.map.ecg_qa import ECGQA
        map_dataset_builder = ECGQA(args, logger)
    elif args.map == "ecg_instruct_45k":
        from ecg_datasets.map.ecg_instruct_45k import ECGInstruct45k
        map_dataset_builder = ECGInstruct45k(args, logger)
    elif args.map == "ecg_bench_pulse":
        from ecg_datasets.map.ecg_bench_pulse import ECGBenchPulse
        map_dataset_builder = ECGBenchPulse(args, logger)
    elif args.map == "ecg_instruct_pulse":
        from ecg_datasets.map.ecg_instruct_pulse import ECGInstructPulse
        map_dataset_builder = ECGInstructPulse(args, logger)

    map_dataset_builder.map_data()