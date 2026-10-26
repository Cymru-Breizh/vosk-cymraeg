#!/usr/bin/env bash


start_time=$(date +%s)


echo "running path.sh"
. ./path.sh
echo "running cmd.sh"
. ./cmd.sh

nj=4       # number of parallel jobs - 1 is perfect for such a small dataset (Was 3 before)
lm_order=3 # language model order (n-gram quantity) - 1 is enough for digits grammar
pocolm_max_memory=4G # memory ceiling passed to pocolm; raise for a larger corpus

# The positional argument is an *upper* bound on the stage -- the opposite of
# the usual Kaldi --stage convention (see howto.md). --start-stage adds the
# missing lower bound, so a range can be run rather than always starting from
# the beginning. That is what lets the CPU stages and the GPU chain stage run
# as two separate jobs (on a cluster, say) against one working directory; left
# at its default the script behaves exactly as before.
start_stage=-1

# Passed through to the chain script, which uses the normal Kaldi convention
# where --stage IS a starting point. Splitting the chain script's own CPU
# stages (0-13, ivector + egs) from its GPU stage (14) would use this.
chain_stage=0

# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1


[[ $# -eq 0 ]] && { echo "Wrong arguments!"; exit 1; }
stage=$1

if [ $stage -eq -1 ]; then
    echo
    echo "===== PREPARING DATA ====="
    echo
    
    # Removing previously created data (from last run.sh execution)
    rm -rf exp mfcc data
    
    # Copying data folder from the workspace output
    echo "Copying data folder"
    cp -r /workspaces/vosk/data/output/ data
    
    # Needs to be prepared by hand (or using self written scripts):
    #
    # spk2gender  [<speaker-id> <gender>]
    # wav.scp     [<uterranceID> <full_path_to_audio_file>]
    # text        [<uterranceID> <text_transcription>]
    # utt2spk     [<uterranceID> <speakerID>]
    # corpus.txt  [<text_transcription>]
   
    # That would be a good place to execute the Python data preparation scripts
    # and copy the generated metadata in the recipe `data` folder
fi

if [ $stage -ge 0 ] && [ $start_stage -le 0 ]; then
    echo
    echo "===== VALIDATING DATA ====="
    echo
    
    for split in train dev test; do
        # Data prepared without a test split (final evaluation outside the recipe)
        [ -d data/$split ] || { echo "Skipping $split (no data/$split directory)"; continue; }
        # Making spk2utt files
        utils/utt2spk_to_spk2utt.pl data/$split/utt2spk > data/$split/spk2utt
        echo
        utils/validate_data_dir.sh --no-feats data/$split
        utils/fix_data_dir.sh data/$split          # tool for data proper sorting if needed - here: for data/train directory
    done
        
    utils/prepare_lang.sh data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
    utils/validate_lang.pl data/lang_nosp
    utils/validate_dict_dir.pl data/local/dict_nosp
    echo "done"
fi

if [ $stage -ge 1 ] && [ $start_stage -le 1 ]; then
    echo
    echo "===== FEATURES EXTRACTION ====="
    echo
    mfccdir=mfcc
    
    for split in train dev test; do
        [ -d data/$split ] || { echo "Skipping $split (no data/$split directory)"; continue; }
        # Making feats.scp files
        steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$split exp/make_mfcc/$split $mfccdir || exit 1;
        # Making cmvn.scp files
        steps/compute_cmvn_stats.sh data/$split exp/make_mfcc/$split $mfccdir || exit 1;
    done
fi


if [ $stage -ge 2 ] && [ $start_stage -le 2 ]; then
    echo
    echo "===== LANGUAGE MODEL CREATION ====="
    echo "==== MAKING lm.arpa (pocolm) ===="
    echo
    if ! command -v train_lm.py >/dev/null; then
        echo "pocolm not found on PATH."
        echo "Expected at \$KALDI_ROOT/tools/pocolm, added to PATH by path.sh."
        echo "Rebuild the training image if it is missing."
        exit 1
    fi

    local=data/local
    mkdir -p $local/tmp $local/lm

    # pocolm consumes a directory of text files: one per data source, plus a
    # held-out dev.txt it uses to tune the discounting constants and -- once
    # there is more than one source -- the interpolation weights. Extra corpora
    # (the techiaith text datasets behind export's --additional_text, or OSCAR)
    # drop in here as additional .txt files, and pocolm learns what each is
    # worth instead of letting raw sentence count decide.
    pocolm_dir=$local/pocolm
    rm -rf $pocolm_dir
    mkdir -p $pocolm_dir/text

    grep -v '^[[:space:]]*$' $local/corpus.txt > $pocolm_dir/text/transcripts.txt

    # dev.txt is the Kaldi dev transcripts with the utterance id stripped. The
    # LM text is built from the train split only (export_kaldi.py), so this is
    # genuinely held out -- measured overlap is ~2%, almost all of it very short
    # filler utterances that recur naturally across splits.
    if [ ! -f data/dev/text ]; then
        echo "data/dev/text is missing -- pocolm needs held-out text to tune on."
        exit 1
    fi
    cut -d' ' -f2- data/dev/text | grep -v '^[[:space:]]*$' > $pocolm_dir/text/dev.txt

    # --num-words=0 leaves the vocabulary unlimited, matching what ngram-count
    # did here, so this swap changes the estimator and nothing else. (pocolm can
    # instead take --wordlist to pin the LM vocab to the lexicon; that moves a
    # second variable, so it belongs in its own experiment.)
    train_lm.py --num-words=0 \
                --num-splits=5 \
                --warm-start-ratio=10 \
                --max-memory=$pocolm_max_memory \
                --keep-int-data=true \
                $pocolm_dir/text $lm_order $pocolm_dir/work $pocolm_dir/lm || exit 1

    format_arpa_lm.py --max-memory=$pocolm_max_memory $pocolm_dir/lm > $local/tmp/lm.arpa || exit 1

    gzip -c $local/tmp/lm.arpa > $local/lm/lm_tglarge.arpa.gz

    # Dev perplexity, so pocolm-vs-SRILM can be compared without a full decode.
    get_data_prob.py --max-memory=$pocolm_max_memory \
                     $pocolm_dir/text/dev.txt $pocolm_dir/lm \
        2>&1 | tee $pocolm_dir/dev_perplexity.log

    echo "done"
fi


if [ $stage -ge 3 ] && [ $start_stage -le 3 ]; then
    echo
    echo "==== MAKING G.fst ===="
    echo
    local=data/local
    lang=data/lang_nosp
    arpa2fst --max-arpa-warnings=-1 --disambig-symbol=#0 \
             --read-symbol-table=$lang/words.txt \
             $local/tmp/lm.arpa \
             $lang/G.fst
    tar -czvf $local/tmp/lm.arpa.gz $local/tmp/lm.arpa
    
    # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
    utils/build_const_arpa_lm.sh $local/lm/lm_tglarge.arpa.gz \
                                 $lang data/lang_nosp_test_tglarge
fi

if [ $stage -ge 4 ] && [ $start_stage -le 4 ]; then
    echo
    echo "===== MONO TRAINING ====="
    echo
    steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
                        data/train data/lang_nosp exp/mono  || exit 1

    echo
    # The decode step is optional
    echo "===== MONO DECODING ====="
    echo
    utils/mkgraph.sh --mono data/lang_nosp exp/mono exp/mono/graph || exit 1
    steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/dev exp/mono/decode
    echo
    echo "===== MONO ALIGNMENT ====="
    echo
    steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" data/train data/lang_nosp exp/mono exp/mono_ali_train || exit 1
fi

if [ $stage -ge 5 ] && [ $start_stage -le 5 ]; then
    echo
    echo "===== TRI1 (first triphone pass) TRAINING ====="
    echo
    # train a first delta + delta-delta triphone system on all utterances
    # Set to 2000 HMM states an 11000 Gaussians
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/train data/lang_nosp exp/mono_ali_train exp/tri1 || exit 1
    echo
    # The decode step is optional
    echo "===== TRI1 (first triphone pass) DECODING ====="
    echo
    utils/mkgraph.sh data/lang_nosp exp/tri1 exp/tri1/graph || exit 1
    steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri1/graph data/dev exp/tri1/decode
    echo
    echo "===== TRI1 ALIGNMENT ====="
    echo
    steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang_nosp exp/tri1 exp/tri1_ali_train
fi

if [ $stage -ge 6 ] && [ $start_stage -le 6 ]; then
    echo
    echo "===== train an LDA+MLLT system ====="
    echo
    steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" 2500 15000 data/train data/lang_nosp exp/tri1_ali_train exp/tri2b
    echo
    echo "===== Align utts using the tri2b model ====="
    echo
    steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true data/train data/lang_nosp exp/tri2b exp/tri2b_ali_train
fi

if [ $stage -ge 7 ] && [ $start_stage -le 7 ]; then
    echo
    echo "===== Train tri3b, which is LDA+MLLT+SAT ====="
    echo
    steps/train_sat.sh --cmd "$train_cmd" 2500 15000 data/train data/lang_nosp exp/tri2b_ali_train exp/tri3b
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -ge 8 ] && [ $start_stage -le 8 ]; then
    echo
    echo "===== compute the pronunciation and silence probabilities ====="
    echo
    steps/get_prons.sh --cmd "$train_cmd" \
        data/train data/lang_nosp exp/tri3b
  
    utils/dict_dir_add_pronprobs.sh --max-normalize true \
        data/local/dict_nosp \
        exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
        exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict
  
    utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang
    utils/validate_lang.pl data/lang
    utils/validate_dict_dir.pl data/local/dict
  
  echo "******* format_lms **********"
  #local/format_lms.sh --src-dir data/lang data/local/lm
  for lm_suffix in tgsmall tgmed; do
      # tglarge is prepared by a separate command, called from run.sh; we don't
      # want to compile G.fst for tglarge, as it takes a while.
      test=data/lang_test_${lm_suffix}
      mkdir -p $test
      cp -r data/lang/* $test
      arpa2fst --disambig-symbol=#0 \
               --read-symbol-table=$test/words.txt \
               data/local/tmp/lm.arpa $test/G.fst
      utils/validate_lang.pl --skip-determinization-check $test || exit 1;
  done
  
  echo "******* build_const_arpa *****"
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
                                 data/lang data/lang_test_tglarge
  
  echo "****** align_fmllr ********"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali_train
fi


if [ $stage -ge 9 ] && [ $start_stage -le 9 ]; then
    echo
    echo "===== STAGE 9 ====="
    echo "==== Test the tri3b system with the silprobs and pron-probs ===="
    echo
    
    # decode using the tri3b model
    utils/mkgraph.sh data/lang_test_tgsmall \
                     exp/tri3b exp/tri3b/graph_tgsmall
    
    echo "****** decode_fmllr.sh ******"
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          exp/tri3b/graph_tgsmall data/dev \
                          exp/tri3b/decode_tgsmall_test
    
    echo "****** lmrescore.sh ******"
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                       data/dev exp/tri3b/decode_{tgsmall,tgmed}_test
    
    echo "****** lmrescore_const_arpa.sh ******"
    steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/dev exp/tri3b/decode_{tgsmall,tglarge}_test
fi


elapsed_time=$(($(date +%s) - start_time))
elapsed_time=$((elapsed_time / 60))
echo "@@@ Elapsed time: $elapsed_time minutes" 


if [ $stage -ge 10 ] && [ $start_stage -le 10 ]; then
    echo
    echo "===== STAGE 10 ====="
    echo "==== Train a chain model ===="
    echo
    
    # Uncomment the desired recipe for training
    #local/chain/run_tdnn_1j_lay12.sh
    local/chain/run_tdnn_260412.sh --stage $chain_stage || { echo "ERROR: chain training (stage 10) failed — model NOT built"; exit 1; }
fi

echo
echo "===== run.sh script is finished ====="
echo
