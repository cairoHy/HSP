#!/bin/bash

RUN_T2T=/code/hsp
# ---------------------------------------settings------------------------------------------
train_gpu=0
eval_gpu=0

use_pos=True
use_pretrain_emb=True
test_data="test"
model="TwoStageSketchV4"  # HSP full model
drop_out_rate=0.2

DATA_PATH=/data/complexwebquestions

MODEL_1=${DATA_PATH}/Models/try-1

# best model setting for demo
best_model_type="TwoStageSketchV4"
BEST_MODEL_1=${DATA_PATH}/Models/try
best_use_pretrain_emb=True
# -----------------------------------------DONT EDIT------------------------------------------------
extra_dim=0
if [ ${use_pos} == "True" ]
then
prepare_args=--shuffle_anno
extra_dim=30
fi
if [ ${use_pretrain_emb} == "True" ]
then
vocab_tail_name="-pre_train"
extra_params=",hidden_size=$((300+${extra_dim})),num_heads=6,filter_size=1200"
fi
if [ ${best_use_pretrain_emb} == "True" ]
then
best_vocab_tail_name="-pre_train"
fi
# ---------------------------------------path settings------------------------------------------
TRAIN_SRC=${DATA_PATH}/train-src.json
TRAIN_TGT=${DATA_PATH}/train-tgt.json
TRAIN_ANNO=${DATA_PATH}/anno/train-src.json.pos
TRAIN_LF=${DATA_PATH}/train.lf
TRAIN_SKETCH=${DATA_PATH}/train.sketch

VALID_FILE=${DATA_PATH}/dev-src.json
REFS_FILE=${DATA_PATH}/dev-tgt.json
REFS_LF=${DATA_PATH}/dev.lf
TRANS_FILE=${VALID_FILE}.trans

ORI_TEST_FILE=${DATA_PATH}/${test_data}-src.json
TEST_FILE=${DATA_PATH}/${test_data}-src.json
TEST_LF=${DATA_PATH}/${test_data}.lf
TEST_REFS_FILE=${DATA_PATH}/${test_data}-tgt.json
TEST_TRANS_FILE=${TEST_FILE}.trans

VOCAB_SRC=${DATA_PATH}/vocab-src${vocab_tail_name}.txt
VOCAB_TGT=${DATA_PATH}/vocab-tgt${vocab_tail_name}.txt

BEST_VOCAB_SRC=${DATA_PATH}/vocab-src${best_vocab_tail_name}.txt
BEST_VOCAB_TGT=${DATA_PATH}/vocab-tgt${best_vocab_tail_name}.txt
# ---------------------------------------pointer-network path settings------------------------------------------
PN_DATA_PATH=${DATA_PATH}/pn
PN_TRAIN_SRC=${PN_DATA_PATH}/train-src.json
PN_TRAIN_TGT=${PN_DATA_PATH}/train-tgt.json
PN_VALID_FILE=${PN_DATA_PATH}/dev-src.json
PN_REFS_FILE=${PN_DATA_PATH}/dev-tgt.json
PN_TEST_FILE=${PN_DATA_PATH}/${test_data}-src.json
PN_TEST_REFS_FILE=${PN_DATA_PATH}/${test_data}-tgt.json
PN_TEST_TRANS_FILE=${PN_TEST_FILE}.trans
# -----------------------------------------Hyper Params------------------------------------------------
TRAIN_PARAMS="device_list=[${train_gpu}],batch_size=128,eval_steps_begin=6000,\
train_steps=20000,save_checkpoint_steps=100,learning_rate=0.4,eval_steps=300,bidirectional=True,\
use_pos=${use_pos},use_pretrained_embedding=${use_pretrain_emb}${extra_params},attention_dropout=${drop_out_rate},\
residual_dropout=${drop_out_rate},relu_dropout=${drop_out_rate}"

DEV_PARAMS="device_list=[${eval_gpu}],decode_batch_size=360,beam_size=1,decode_alpha=0.6"

TEST_PARAMS="device_list=[${eval_gpu}],decode_batch_size=160,beam_size=16,decode_alpha=0.6"

INTERFACE_PARAMS="device_list=[0],decode_batch_size=1,beam_size=16,decode_alpha=0.6"
# -----------------------------------------------------------------------------------------
MODE=$1

if [ ${MODE} == "preprocess" ]
then

echo "Preprocess Dataset......"
python3 ${RUN_T2T}/preprocess/gen_data.py --target_data_path ${DATA_PATH} \
--ori_complex_data_path ${DATA_PATH}  --ori_data_path ${DATA_PATH}
if [ ${use_pos} == "True" ]
then

python3 ${RUN_T2T}/preprocess/annotate_sequence.py --prefix ${DATA_PATH}
fi

elif [ ${MODE} == "prepare" ]
then

echo "Shuffle Dataset........"
python3 ${RUN_T2T}/prepare/shuffle_dataset.py --input ${TRAIN_SRC} ${TRAIN_TGT} \
${TRAIN_ANNO} ${TRAIN_LF} ${TRAIN_SKETCH} \
${prepare_args}
cat ${TRAIN_SRC}.shuf ${TRAIN_TGT}.shuf ${TRAIN_LF} ${TRAIN_SKETCH} ${TRAIN_SKETCH} ${TRAIN_SKETCH} > ${TRAIN_SRC}.whole
echo "Build Vocab........"
python3 ${RUN_T2T}/prepare/build_vocab.py ${TRAIN_SRC}.whole ${VOCAB_SRC} --vocabsize 37000 --threshold 4 \
    --use_pretrain_emb ${use_pretrain_emb}
echo "Build LogicForm Vocab........"
cp ${VOCAB_SRC} ${DATA_PATH}/vocab-lf.txt
echo "Build Sketch Vocab........"
cp ${VOCAB_SRC} ${DATA_PATH}/vocab-sketch.txt

cp ${VOCAB_SRC} ${VOCAB_TGT}
rm ${TRAIN_SRC}.whole

elif [ ${MODE} == "train" ]
then

echo "Start Training........"
python3 ${RUN_T2T}/train.py --input ${TRAIN_SRC}.shuf ${TRAIN_TGT}.shuf \
    --output ${MODEL_1} \
    --vocab ${VOCAB_SRC} ${VOCAB_TGT} \
    --validation ${VALID_FILE} \
    --references ${REFS_FILE} \
    --parameters=${TRAIN_PARAMS} \
    --dev_params=${TEST_PARAMS} \
    --model ${model}

elif [ ${MODE} == "train_lf" ]
then

echo "Start Training to logic_form........"
python3 ${RUN_T2T}/train.py --input ${TRAIN_SRC}.shuf ${TRAIN_LF}.shuf \
    --output ${MODEL_1} \
    --vocab ${DATA_PATH}/vocab-lf.txt ${DATA_PATH}/vocab-lf.txt \
    --validation ${VALID_FILE} \
    --references ${REFS_LF} \
    --parameters=${TRAIN_PARAMS} \
    --dev_params=${DEV_PARAMS} \
    --metric em \
    --model ${model}

elif [ ${MODE} == "train_pn" ]
then

echo "Start Training to logic_form [pointer-network] ........"
python3 ${RUN_T2T}/train.py --input ${PN_TRAIN_SRC} ${PN_TRAIN_TGT} \
    --output ${MODEL_1} \
    --vocab ${DATA_PATH}/vocab-lf.txt ${DATA_PATH}/vocab-lf.txt \
    --validation ${PN_VALID_FILE} \
    --references ${PN_REFS_FILE} \
    --parameters=${TRAIN_PARAMS} \
    --dev_params=${TEST_PARAMS} \
    --metric em \
    --model ${model}

elif [ ${MODE} == "test" ]
then

echo "Start Testing........."
python3 ${RUN_T2T}/inference.py --input ${TEST_FILE} \
    --output ${TEST_TRANS_FILE} \
    --vocab ${VOCAB_SRC} ${VOCAB_TGT} \
    --models ${MODEL_1}/best \
    --parameters=${TEST_PARAMS} \
    --model ${model}
python3 ${RUN_T2T}/evaluate.py --prefix ${DATA_PATH} \
    --ori_file ${TEST_FILE} \
    --pred_file ${TEST_TRANS_FILE} \
    --ref_file ${TEST_REFS_FILE}

elif [ ${MODE} == "test_lf" ]
then

echo "Start Testing logic_form........."
python3 ${RUN_T2T}/inference.py --input ${TEST_FILE} \
    --output ${TEST_TRANS_FILE} \
    --vocab ${DATA_PATH}/vocab-lf.txt ${DATA_PATH}/vocab-lf.txt \
    --models ${MODEL_1}/best \
    --parameters=${TEST_PARAMS} \
    --model ${model}
python3 ${RUN_T2T}/evaluate.py --prefix ${DATA_PATH} \
    --ori_file ${TEST_FILE} \
    --pred_file ${TEST_TRANS_FILE} \
    --ref_file ${TEST_LF}

elif [ ${MODE} == "test_pn" ]
then

echo "Start Testing logic_form [pointer network] ........."
python3 ${RUN_T2T}/inference.py --input ${PN_TEST_FILE} \
    --output ${PN_TEST_TRANS_FILE} \
    --vocab ${DATA_PATH}/vocab-lf.txt ${DATA_PATH}/vocab-lf.txt \
    --models ${MODEL_1}/best \
    --parameters=${TEST_PARAMS} \
    --model ${model}
python3 ${RUN_T2T}/evaluate.py --prefix ${DATA_PATH} \
    --ori_file ${PN_TEST_FILE} \
    --pred_file ${PN_TEST_TRANS_FILE} \
    --ref_file ${PN_TEST_REFS_FILE}

elif [ ${MODE} == "test_ori" ]
then

echo "Start Eval origin model performance........."
python3 ${RUN_T2T}/evaluate.py --prefix ${DATA_PATH} \
    --ori_file ${ORI_TEST_FILE} \
    --pred_file ${TEST_TRANS_FILE} \
    --ref_file ${TEST_REFS_FILE} \
    --calcu_origin