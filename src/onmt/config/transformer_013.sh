#================================================== PREPROCESS ==================================================
!onmt_preprocess -train_src '$PROCESS/train/squad_id_split0.9_uncased_source.txt' -train_tgt '$PROCESS/train/squad_id_split0.9_uncased_target.txt' \
    -valid_src '$PROCESS/val/squad_id_split0.9_uncased_source.txt' -valid_tgt '$PROCESS/val/squad_id_split0.9_uncased_target.txt' \
    -save_data '$SAVEFILE/squad_id_split0.9_uncased_copy' \
    -overwrite \
    -dynamic_dict \
    -src_vocab_size 50000 \
    -tgt_vocab_size 30000 \
    -src_seq_length 60 \
    -tgt_seq_length 20

#================================================== EMBEDDING ==================================================
!python '$SRCFILE/embeddings_to_torch.py' -emb_file_both '$WE/ft_to_gl_300_id.vec' \
    -dict_file '$SAVEFILE/squad_id_split0.9_uncased_copy.vocab.pt' \
    -output_file '$SAVEFILE/embeddings_uncased_copy'

#================================================== TRAIN ==================================================
!onmt_train -data '$SAVEFILE/squad_id_split0.9_uncased_copy' -save_model '$CHKP/onmt/transformer_013' \
    -world_size 1 -gpu_ranks 0 \
    -seed 42 \
    -pre_word_vecs_enc '$SAVEFILE/embeddings_uncased_copy.enc.pt' \
    -pre_word_vecs_dec '$SAVEFILE/embeddings_uncased_copy.dec.pt' \
    -fix_word_vecs_enc \
    -fix_word_vecs_dec \
    -feat_merge mlp \
    -keep_checkpoint 5 \
    -layers 2 -rnn_size 300 -word_vec_size 300 -transformer_ff 256 -heads 2 \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 120600  -max_generator_batches 2 -dropout 0.1 \
    -batch_size 256 -batch_type tokens -normalization tokens -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4020 -learning_rate 1 \
    -max_grad_norm 1 -param_init 0  -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 4020 -save_checkpoint_steps 4020 \
    -copy_attn \
    -copy_attn_force

#================================================== TRANSLATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
!onmt_translate -model '$CHKP/onmt/transformer_013_step_120600.pt' \
    -src '$TESTFILE/squad_id_split0.9_uncased_source.txt' -output '$MODEL/reports/txts/onmt/transformer_013_step_120600_pred.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20
#-------------------------------------------------- TYDIQA --------------------------------------------------
!onmt_translate -model '$CHKP/onmt/transformer_013_step_120600.pt' \
    -src '$TESTFILE/tydiqa_id_split0.9_uncased_source.txt' -output '$MODEL/reports/txts/onmt/transformer_013_step_120600_pred_tydiqa.txt' -replace_unk \
    -seed 42 \
    -beam_size 5 \
    -max_length 20

#================================================== EVALUATE ==================================================
#-------------------------------------------------- SQUAD --------------------------------------------------
!python '$SRCFILE/run_evaluation.py' \
    --source_file='$TESTFILE/squad_id_split0.9_uncased_source.txt' \
    --target_file='$TESTFILE/squad_id_split0.9_uncased_target.txt' \
    --prediction_file='$MODEL/reports/txts/onmt/transformer_013_step_120600_pred.txt' \
    --log_file='$MODEL/reports/txts/onmt/eval_log_transformer_013_step_120600.txt'

#-------------------------------------------------- TYDIQA --------------------------------------------------
!python '$SRCFILE/run_evaluation.py' \
    --source_file='$TESTFILE/tydiqa_id_split0.9_uncased_source.txt' \
    --target_file='$TESTFILE/tydiqa_id_split0.9_uncased_target.txt' \
    --prediction_file='$MODEL/reports/txts/onmt/transformer_013_step_120600_pred_tydiqa.txt' \
    --log_file='$MODEL/reports/txts/onmt/eval_log_transformer_013_step_120600_tydiqa.txt'
