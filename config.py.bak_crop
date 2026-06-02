import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Train & Dev Data folder prepare
    parser.add_argument("--atadd_t1_train_audio", type=str, help="Path to the training audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/train')
    parser.add_argument("--atadd_t1_train_label", type=str, help="Path to the training label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/train.csv")
    parser.add_argument("--atadd_t1_dev_audio", type=str, help="Path to the development audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/dev')
    parser.add_argument("--atadd_t1_dev_label", type=str, help="Path to the development label for ATADD T1 dataset",
                        default="yourpath/atadd/T1/label/dev.csv")
    parser.add_argument("--atadd_t1_eval_audio", type=str, help="Path to the evaluation audio for ATADD T1 dataset",
                        default='yourpath/atadd/T1/eval')


    parser.add_argument("--atadd_t2_train_audio", type=str, help="Path to the training audio for ATADD T2 dataset",
                        default='AT_ADD_data/Track2/train')
    parser.add_argument("--atadd_t2_train_label", type=str, help="Path to the training label for ATADD T2 dataset",
                        default="AT_ADD_data/Track2/label/train.csv")
    parser.add_argument("--atadd_t2_dev_audio", type=str, help="Path to the development audio for ATADD T2 dataset",
                        default='AT_ADD_data/Track2/dev')
    parser.add_argument("--atadd_t2_dev_label", type=str, help="Path to the development label for ATADD T2 dataset",
                        default="AT_ADD_data/Track2/label/dev.csv")
    parser.add_argument("--atadd_t2_eval_audio", type=str, help="Path to the evaluation audio for ATADD T2 dataset",
                        default='AT_ADD_data/Track2/eval_progress')


    # SSL folder prepare
    parser.add_argument("--xlsr", default="huggingface/wav2vec2-xls-r-300m")
    parser.add_argument("--wavlm", default="huggingface/wavlm-large/")
    parser.add_argument("--mert", default="huggingface/MERT-v1-330M/")

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    # countermeasure
    parser.add_argument("--audio_len", type=int, help="raw waveform length", default=64600)
    parser.add_argument('-m', '--model', help='Model arch', default='pt-w2v2aasist',
                        choices=[
                            'specresnet', 'aasist',
                            'ft-w2v2aasist', 'fr-wavlmaasist', 'fr-mertaasist',
                            'fr-w2v2aasist', 'ft-wavlmaasist', 'ft-mertaasist',
                            'pt-w2v2aasist', 'wpt-w2v2aasist',
                            'pt-wavlmaasist', 'wpt-wavlmaasist',
                            'pt-mertaasist', 'wpt-mertaasist',
                            't2-router-xlsr-mert',
                            'ufaformer-full', 'ufm-track2-full',])

    # pt
    parser.add_argument("--prompt_dim", type=int, help="prompt dim", default=1024)
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=10)
    parser.add_argument("--pt_dropout", type=float, help="dropout", default=0.1)

    # wpt
    parser.add_argument("--num_wavelet_tokens", type=int, help="wavelet token", default=4)
    
    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    # =========================
    # Track2 type-aware training
    # =========================
    parser.add_argument(
        "--t2_return_type",
        action="store_true",
        help="Return Track2 audio type id: speech/sound/singing/music"
    )

    parser.add_argument(
        "--t2_gdro",
        action="store_true",
        help="Use Track2 type-balanced GroupDRO loss"
    )

    parser.add_argument(
        "--t2_gdro_eta",
        type=float,
        default=2.0,
        help="Softmax temperature for GroupDRO type weights"
    )

    parser.add_argument(
        "--t2_type_adv",
        action="store_true",
        help="Use type-adversarial learning with gradient reversal"
    )

    parser.add_argument(
        "--t2_type_adv_weight",
        type=float,
        default=0.05,
        help="Weight of type adversarial loss"
    )

    parser.add_argument(
        "--t2_grl_lambda",
        type=float,
        default=1.0,
        help="Gradient reversal strength"
    )

    parser.add_argument(
        "--t2_type_feat_dim",
        type=int,
        default=160,
        help="Feature dim for SSLAASIST last_hidden. Default 160."
    )

    # =========================
    # Track2 XLSR+MERT routed expert
    # =========================
    parser.add_argument(
        "--t2_router_freeze_xlsr",
        action="store_true",
        help="Freeze XLSR expert in type-routed model"
    )

    parser.add_argument(
        "--t2_router_freeze_mert",
        action="store_true",
        help="Freeze MERT expert in type-routed model"
    )

    parser.add_argument(
        "--t2_router_type_loss",
        type=float,
        default=0.0,
        help="Auxiliary type classification loss weight for router"
    )

    parser.add_argument(
        "--t2_router_entropy",
        type=float,
        default=0.0,
        help="Entropy regularization weight for expert router"
    )

    ####5.13 修改 T2-GDRO-ADV + T2-Router-XLSR-MERT
    
    # =========================
    # UFA-Former-Full
    # =========================
    parser.add_argument("--obeats", default="huggingface/OpenBEATs-ICME")

    parser.add_argument("--ufa_freeze_xlsr", action="store_true")
    parser.add_argument("--ufa_freeze_mert", action="store_true")
    parser.add_argument("--ufa_freeze_obeats", action="store_true")

    parser.add_argument("--ufa_hidden_dim", type=int, default=1024)
    parser.add_argument("--ufa_router_hidden", type=int, default=256)
    parser.add_argument("--ufa_num_heads", type=int, default=8)
    parser.add_argument("--ufa_num_layers", type=int, default=2)
    parser.add_argument("--ufa_dropout", type=float, default=0.1)

    parser.add_argument("--ufa_type_loss", type=float, default=0.1)
    parser.add_argument("--ufa_router_entropy", type=float, default=0.01)

    parser.add_argument("--ufa_stft_n_fft", type=int, default=1024)
    parser.add_argument("--ufa_stft_hop", type=int, default=320)

    parser.add_argument("--init_from", type=str, default="")
    
    # =========================
    # UFM-Track2-Full
    # =========================
    parser.add_argument(
        "--beats",
        default="/root/autodl-tmp/AT-ADD-Baseline-track2-R2/huggingface/OpenBEATs-ICME",
        help="Path to BEATs/OpenBEATs/general-audio SSL model"
    )

    parser.add_argument(
        "--ufm_dim",
        type=int,
        default=512,
        help="Hidden dimension of UFM streams"
    )

    parser.add_argument(
        "--ufm_mem_slots",
        type=int,
        default=16,
        help="Number of memory slots for each forgery memory bank"
    )

    parser.add_argument(
        "--ufm_heads",
        type=int,
        default=8,
        help="Number of attention heads in UFM cross-stream transformer"
    )

    parser.add_argument(
        "--ufm_layers",
        type=int,
        default=2,
        help="Number of cross-stream transformer layers"
    )

    parser.add_argument(
        "--ufm_dropout",
        type=float,
        default=0.1,
        help="Dropout in UFM modules"
    )

    parser.add_argument(
        "--ufm_freeze_xlsr",
        action="store_true",
        help="Freeze XLSR expert"
    )

    parser.add_argument(
        "--ufm_freeze_mert",
        action="store_true",
        help="Freeze MERT expert"
    )

    parser.add_argument(
        "--ufm_freeze_beats",
        action="store_true",
        help="Freeze BEATs/OpenBEATs expert"
    )

    parser.add_argument(
        "--ufm_type_loss",
        type=float,
        default=0.1,
        help="Auxiliary type classification loss weight"
    )

    parser.add_argument(
        "--ufm_router_entropy",
        type=float,
        default=0.01,
        help="Router entropy regularization weight"
    )
    return parser
