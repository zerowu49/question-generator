#================================================== PREPROCESS ==================================================
!onmt_preprocess -train_src '$PROCESS/train/squad_id_split0.9_uncased_source.txt' -train_tgt '$PROCESS/train/squad_id_split0.9_uncased_target.txt' \
    -valid_src '$PROCESS/val/squad_id_split0.9_uncased_source.txt' -valid_tgt '$PROCESS/val/squad_id_split0.9_uncased_target.txt' \
    -save_data '$SAVEFILE/squad_id_split0.9_uncased' \
    -overwrite \
    -dynamic_dict \
    -src_vocab_size 50000 \
    -tgt_vocab_size 30000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
!python '$SRCFILE/embeddings_to_torch.py' -emb_file_both '$WE/ft_to_gl_300_id.vec' \
    -dict_file '$SAVEFILE/squad_id_split0.9_uncased.vocab.pt' \
    -output_file '$SAVEFILE/embeddings_uncased'

#================================================== TRAIN ==================================================
!onmt_train -data '$SAVEFILE/squad_id_split0.9_uncased' -save_model '$CHKP/onmt/gru_041' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -save_checkpoint_steps 8025 \
    -word_vec_size 300 \
    -pre_word_vecs_enc '$SAVEFILE/embeddings_uncased.enc.pt' \
    -pre_word_vecs_dec '$SAVEFILE/embeddings_uncased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -keep_checkpoint 2 \
    -optim 'adam' \
    -learning_rate 0.001 \
    -learning_rate_decay 0.95 \
    -start_decay_steps 16050 \
    -rnn_type GRU \
    -encoder_type brnn \
    -layers 2 \
    -global_attention mlp \
    -rnn_size 256 \
    -train_steps 32100 \
    -valid_steps 3210 \
    -batch_size 64 \
    -dropout 0.3

#================================================== TRANSLATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
!onmt_translate -model '$CHKP/onmt/gru_041_step_32100.pt' \
    -src '$TESTFILE/squad_id_split0.9_uncased_source.txt' -output '$MODEL/reports/txts/onmt/gru_041_step_32100_pred.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#-------------------------------------------------- TYDIQA --------------------------------------------------
!onmt_translate -model '$CHKP/onmt/gru_041_step_32100.pt' \
    -src '$TESTFILE/tydiqa_id_split0.9_uncased_source.txt' -output '$MODEL/reports/txts/onmt/gru_041_step_32100_pred_tydiqa.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
!python '$SRCFILE/run_evaluation.py' \
    --source_file='$TESTFILE/squad_id_split0.9_uncased_source.txt' \
    --target_file='$TESTFILE/squad_id_split0.9_uncased_target.txt' \
    --prediction_file='$MODEL/reports/txts/onmt/gru_041_step_32100_pred.txt' \
    --log_file='$MODEL/reports/txts/onmt/eval_log_gru_041_step_32100.txt'

#-------------------------------------------------- TYDIQA --------------------------------------------------
!python '$SRCFILE/run_evaluation.py' \
    --source_file='$TESTFILE/tydiqa_id_split0.9_uncased_source.txt' \
    --target_file='$TESTFILE/tydiqa_id_split0.9_uncased_target.txt' \
    --prediction_file='$MODEL/reports/txts/onmt/gru_041_step_32100_pred_tydiqa.txt' \
    --log_file='$MODEL/reports/txts/onmt/eval_log_gru_041_step_32100_tydiqa.txt'
