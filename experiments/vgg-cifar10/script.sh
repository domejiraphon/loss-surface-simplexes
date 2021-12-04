
python base_trainer_svhn.py -model_dir base/lenet/pf0.5 -tensorboard -pf 0.5 -lenet -restart
python simplex_trainer_all.py -model_dir simplex/bad_mode_good_con -load_model base/pf0.5/100.pt -tensorboard -pf 0