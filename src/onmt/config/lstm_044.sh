#================================================== PREPROCESS ==================================================
!onmt_preprocess -train_src '$PROCESS/train/squad_id_split0.9_cased_source.txt' -train_tgt '$PROCESS/train/squad_id_split0.9_cased_target.txt' \
    -valid_src '$PROCESS/val/squad_id_split0.9_cased_source.txt' -valid_tgt '$PROCESS/val/squad_id_split0.9_cased_target.txt' \
    -save_data '$SAVEFILE/squad_id_split0.9_cased' \
    -overwrite \
    -dynamic_dict \
    -src_vocab_size 50000 \
    -tgt_vocab_size 30000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
!python '$SRCFILE/embeddings_to_torch.py' -emb_file_both '$WE/ft_to_gl_300_id.vec' \
    -dict_file '$SAVEFILE/squad_id_split0.9_cased.vocab.pt' \
    -output_file '$SAVEFILE/embeddings_cased'

#================================================== TRAIN ==================================================
!onmt_train -data '$SAVEFILE/squad_id_split0.9_cased' -save_model '$CHKP/onmt/lstm_044' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -save_checkpoint_steps 8025 \
    -word_vec_size 300 \
    -pre_word_vecs_enc '$SAVEFILE/embeddings_cased.enc.pt' \
    -pre_word_vecs_dec '$SAVEFILE/embeddings_cased.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -keep_checkpoint 3 \
    -optim 'adam' \
    -learning_rate 0.001 \
    -learning_rate_decay 0.5 \
    -start_decay_steps 8025 \
    -rnn_type LSTM \
    -encoder_type brnn \
    -layers 2 \
    -global_attention general \
    -rnn_size 600 \
    -train_steps 16050 \
    -valid_steps 3210 \
    -batch_size 64 \
    -dropout 0.3

#================================================== TRANSLATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
!onmt_translate -model '$CHKP/onmt/lstm_044_step_16050.pt' \
    -src '$TESTFILE/squad_id_split0.9_cased_source.txt' -output '$MODEL/reports/txts/onmt/lstm_044_step_16050_pred.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#-------------------------------------------------- TYDIQA --------------------------------------------------
!onmt_translate -model '$CHKP/onmt/lstm_044_step_16050.pt' \
    -src '$TESTFILE/tydiqa_id_split0.9_cased_source.txt' -output '$MODEL/reports/txts/onmt/lstm_044_step_16050_pred_tydiqa.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
!python '$SRCFILE/run_evaluation.py' \
    --source_file='$TESTFILE/squad_id_split0.9_cased_source.txt' \
    --target_file='$TESTFILE/squad_id_split0.9_cased_target.txt' \
    --prediction_file='$MODEL/reports/txts/onmt/lstm_044_step_16050_pred.txt' \
    --log_file='$MODEL/reports/txts/onmt/eval_log_lstm_044_step_16050.txt'

#-------------------------------------------------- TYDIQA --------------------------------------------------
!python '$SRCFILE/run_evaluation.py' \
    --source_file='$TESTFILE/tydiqa_id_split0.9_cased_source.txt' \
    --target_file='$TESTFILE/tydiqa_id_split0.9_cased_target.txt' \
    --prediction_file='$MODEL/reports/txts/onmt/lstm_044_step_16050_pred_tydiqa.txt' \
    --log_file='$MODEL/reports/txts/onmt/eval_log_lstm_044_step_16050_tydiqa.txt'
